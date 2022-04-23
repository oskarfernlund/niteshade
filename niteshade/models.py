#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Abstract base model class for niteshade workflows as well as some specific toy 
models for out-of-the-box use.
"""

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from niteshade.data import DataLoader


# =============================================================================
#  CLASSES
# =============================================================================

class BaseModel(nn.Module):
    """Abstract model class intended for ease of implementation in designing 
       neural networks for data poisining attacks. Requires an architecture
       to be defined in the form of a list or nested list containing the 
       sequence of torch.nn.modules objects needed to perform a forward 
       pass. 
    """
    def __init__(self, architecture: list, optimizer: str, 
                 loss_func: str, lr: float, optim_kwargs = {}, 
                 loss_kwargs = {}, seed = None):
        """
        Constrcutor method of BaseModel class that inherits from nn.Module.
        Args: 
            architecture (list) : list or nested list containing sequence of 
                                  nn.torch.modules objects to be used in the 
                                  forward pass of the model.
            optimizer (str) : String specifying optimizer to use in training neural network.
                              Options:
                                'adam': torch.optim.Adam(),
                                'adagrad': torch.optim.Adagrad(),
                                'adamax': torch.optim.Adamax(),
                                'sgd': torch.optim.SGD().
            loss_func (str) : String specifying loss function to use in training neural network.
                              Options:
                                'mse': nn.MSELoss(),
                                'nll': nn.NLLLoss(),
                                'bce': nn.BCELoss(),
                                'cross_entropy': nn.CrossEntropyLoss().
            lr (float) : Learning rate to use in training neural network.
            optim_kwargs (dict) : dictionary containing additional optimizer key-word 
                                  arguments (Default = {}).
            loss_kwargs (dict) : dictionary containing additional key-word arguments 
                                 for the loss function (Default = {}).
        """
        super().__init__()
        #initialise attributes to store training hyperparameters
        self.lr = lr
        self.loss_func_str = loss_func
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #retrieve user-defined sequences of layers
        if any(isinstance(el, list) for el in architecture):
            self.network = nn.ModuleList(nn.Sequential(*seq) for seq in architecture)
        else: 
            self.network = nn.Sequential(*architecture)
        
        #apply random seed
        if seed: 
            torch.manual_seed(seed)

        if optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **optim_kwargs)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, **optim_kwargs)
        elif optimizer == "adagrad": 
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr, **optim_kwargs)
        elif optimizer == "adamax":
            self.optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr, **optim_kwargs)
        else: 
            raise NotImplementedError(f"The optimizer {optimizer} has not been implemented.")
        
        if loss_func.lower() == "mse":
            self.loss_func = nn.MSELoss(**loss_kwargs)
        elif loss_func.lower() == "cross_entropy":
            self.loss_func = nn.CrossEntropyLoss(**loss_kwargs)
        elif loss_func.lower() == "nll":
            self.loss_func = nn.NLLLoss(**loss_kwargs)  
        elif loss_func.lower() == "bce":
            self.loss_func = nn.BCELoss(**loss_kwargs)      
        else: 
            raise NotImplementedError(f"The loss function {loss_func} has not been implemented.")

        self.losses = []
    
    def _check_inputs(self, X, y):
        assert (isinstance(X, (np.ndarray, torch.Tensor)) 
                and isinstance(y, (np.ndarray, torch.Tensor)))

        #convert np.ndarray /pd.Dataframe to tensor for the NN
        if (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            X = torch.tensor(X, dtype=torch.float64)
            if self.loss_func_str in ['mse']:
                y = torch.tensor(y, dtype=torch.float64)
            if self.loss_func_str in ['nll', 'bce', 'cross_entropy']:
                #check if one-hot encoded
                if len(y.shape) > 1: 
                    y = torch.tensor(y).argmax(dim=1)
                else: 
                    y = torch.tensor(y, dtype=torch.long)
        else:
            X = X.type(torch.float64)
            if self.loss_func_str in ['mse']:
                y = y.type(torch.float64)
            else:
                if len(y.shape) > 1: 
                    y = y.argmax(dim=1)
                else: 
                    y = y.type(torch.long)

        return X, y
    
    def step(self, X_batch, y_batch):
        """Perform a step of gradient descent on the passed inputs (X_batch) and labels (y_batch).

        Args:
             X_batch (np.ndarray, torch.Tensor) : input data used in training.
             y_batch (np.ndarray, torch.Tensor) : target data used in training.
        """
        X_batch, y_batch = self._check_inputs(X_batch, y_batch)

        self.train() #set model in training mode

        #send data to device
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        #zero gradients so they are not accumulated across batches
        self.optimizer.zero_grad()

        # Performs forward pass through classifier
        outputs = self.forward(X_batch.float())

        # Computes loss on batch with given loss function
        loss = self.loss_func(outputs, y_batch)
        self.losses.append(loss)

        # Performs backward pass through gradient of loss wrt model parameters
        loss.backward()

        #update model parameters after gradients are updated
        self.optimizer.step()
    
    def forward(self, x):
        """Perform a forward pass through the regressor.
        
        Args:
            x (torch.Tensor) : Processed input array of size (batch_size, input_size).
            
        Returns: 
            Predictions from current state of the model.
        """
        raise NotImplementedError

    def predict(self, x):
        """Predict on a data sample (x).

        Args: 
            x (np.ndarray, torch.Tensor) : sample to predict from. 
        """
        raise NotImplementedError

    def evaluate(self, X_test, y_test, batch_size):
        """Test the accuracy of the iris classifier on a test set.

        Args:
            X_test (np.ndarray, torch.Tensor) : test input data.
            y_test (np.ndarray, torch.Tensor) : test target data.
            batch_size (int) : size of batches in DataLoader object.
        """
        raise NotImplementedError
        
#====================================================
#=======================IRIS========================
#====================================================
class IrisClassifier(BaseModel):
    """Pre-defined simple classifier for the Iris dataset containing 
       one fully-connected layer with 16 neurons using ReLU. 
    """
    def __init__(self, optimizer="adam", loss_func="cross_entropy", lr=0.001,
                 optim_kwargs = {}, loss_kwargs = {}):
        """Construct network as per user specifications.

        Args:
            optimizer (str) : String specifying optimizer to use in training neural network (Default = 'adam').
                              Options:
                                'adam': torch.optim.Adam(),
                                'adagrad': torch.optim.Adagrad(),
                                'sgd': torch.optim.SGD().
            loss_func (str) : String specifying loss function to use in training neural network (Default = 'cross_entropy').
                              Options:
                                'mse': nn.MSELoss(),
                                'nll': nn.NLLLoss(),
                                'bce': nn.BCELoss(),
                                'cross_entropy': nn.CrossEntropyLoss().
            lr (float) : Learning rate to use in training neural network (Default = 0.001).
            optim_kwargs (dict) : dictionary containing additional optimizer key-word 
                                  arguments (Default = {}).
            loss_kwargs (dict) : dictionary containing additional key-word arguments 
                                 for the loss function (Default = {}).
        """
        #pre-defined simple architecture for classification on the iris dataset
        architecture = [nn.Linear(4, 50), 
                        nn.ReLU(), 
                        nn.Linear(50,50),
                        nn.ReLU(),
                        nn.Linear(50, 3), 
                        nn.Softmax()]

        super().__init__(architecture, optimizer, loss_func, lr, optim_kwargs, loss_kwargs)
    
    def forward(self, x):
        "Forward method for model (needed as subclass of nn.Module)."
        return self.network(x) 

    def predict(self, x):
        """Predict on a data sample."""
        with torch.no_grad():
            pred =  self.forward(x)
        return pred

    def evaluate(self, X_test, y_test, batch_size):
        """Test the accuracy of the iris classifier on a test set.

        Args:
            X_test (np.ndarray) : test input data.
            y_test (np.ndarray) : test target data.
            batch_size (int) : size of batches in DataLoader object.
        """
        X_test, y_test = self._check_inputs(X_test, y_test)

        #create dataloader with test data
        test_loader = DataLoader(X_test, y_test, batch_size=batch_size)
        num_batches = len(test_loader)

        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        with torch.no_grad():
            test_loss = 0
            correct = 0

            for inputs, targets in test_loader:
                outputs = self.forward(inputs.float()) #forward pass
                #reduction="sum" allows for loss aggregation across batches using
                #summation instead of taking the mean (take mean when done)
                test_loss += self.loss_func(outputs, targets).item()

                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum()

        num_points = batch_size * num_batches

        test_loss /= num_points #mean loss

        accuracy = correct / num_points
        
        return accuracy
    
#====================================================
#=======================MNIST========================
#====================================================
class MNISTClassifier(BaseModel):
    """
    Pre-defined classifier for the MNIST dataset.

    Args:
        optimizer (str) : String specifying optimizer to use in training neural network (Default = 'sgd').
                            Options:
                            'adam': torch.optim.Adam(),
                            'adagrad': torch.optim.Adagrad(),
                            'sgd': torch.optim.SGD()
        loss_func (str) : String specifying loss function to use in training neural network (Default = 'nll').
                            Options:
                            'mse': nn.MSELoss(),
                            'nll': nn.NLLLoss(),
                            'bce': nn.BCELoss(),
                            'cross_entropy': nn.CrossEntropyLoss().
        lr (float) : Learning rate to use in training neural network (Default = 0.01).
        optim_kwargs (dict) : dictionary containing additional optimizer key-word 
                              arguments (Default = {}).
        loss_kwargs (dict) : dictionary containing additional key-word arguments 
                             for the loss function (Default = {}).
    """
    def __init__(self, optimizer="sgd", loss_func="nll", lr=0.01):
        #pre-defined architecture for classification on the MNIST dataset
        conv_layers = [nn.Conv2d(1, 10, kernel_size=5),
                       nn.MaxPool2d(kernel_size=2), 
                       nn.ReLU(), 
                       nn.Conv2d(10, 20, kernel_size=5), 
                       nn.Dropout2d(), 
                       nn.MaxPool2d(kernel_size=2), 
                       nn.ReLU(), 
                      ]
        dense_layers = [nn.Linear(320, 50), 
                        nn.ReLU(), 
                        nn.Linear(50, 10), 
                        nn.LogSoftmax(dim=-1)
                        ]

        architecture = [conv_layers, dense_layers]
                        
        super().__init__(architecture, optimizer, loss_func, lr)
    
    def forward(self, x):
        "Forward method for model (needed as subclass of nn.Module)."
        #unpack network sequences
        conv_sequential, dense_sequential = self.network 
        
        #forward pass through model
        x = conv_sequential(x)
        x = x.view(-1, 320)
        x = dense_sequential(x)

        return x

    def predict(self, x):
        """Predict on a data sample."""
        self.eval() #set model in eval mode
        with torch.no_grad():
            pred =  self.forward(x)
        return pred

    def evaluate(self, X_test, y_test, batch_size):
        """Test the accuracy of the iris classifier on a test set.

        Args:
            X_test (np.ndarray) : test input data.
            y_test (np.ndarray) : test target data.
        """
        X_test, y_test = self._check_inputs(X_test, y_test)

        self.eval()
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        #create dataloader with test data
        test_loader = DataLoader(X_test, y_test, batch_size=batch_size)
        num_batches = len(test_loader)

        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                output = self.forward(inputs.float())
                test_loss += self.loss_func(output, targets).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.view_as(pred)).sum()

        num_points = num_batches * batch_size
        test_loss /= num_points
        accuracy = 100. * correct / num_points
        
        return accuracy

#====================================================
#=======================CIFAR10======================
#====================================================
class _ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CifarClassifier(BaseModel):
    """
    ResNet-18 classifier inheriting from BaseModel for the torchvision 
    CIFAR10 dataset. Achieves 78% accuracy on the held out test set in 20
    epochs using mini-batches of size 32.

    Args:
        optimizer (str) : String specifying optimizer to use in training neural network (Default = 'adam').
                            Options:
                            'adam': torch.optim.Adam(),
                            'adagrad': torch.optim.Adagrad(),
                            'sgd': torch.optim.SGD()
        loss_func (str) : String specifying loss function to use in training neural network (Default = 'cross_entropy').
                            Options:
                            'mse': nn.MSELoss(),
                            'nll': nn.NLLLoss(),
                            'bce': nn.BCELoss(),
                            'cross_entropy': nn.CrossEntropyLoss().
        lr (float) : Learning rate to use in training neural network (Default = 0.0001).
        optim_kwargs (dict) : dictionary containing additional optimizer key-word 
                              arguments (Default = {}).
        loss_kwargs (dict) : dictionary containing additional key-word arguments 
                             for the loss function (Default = {}).
    """
    def __init__(self, optimizer="adam", loss_func="cross_entropy", lr=0.0001,
                 optim_kwargs = {'weight_decay': 1e-6}, loss_kwargs = {}):

        self.inchannel = 64
        conv1 = [nn.Conv2d(3, 64, kernel_size = 3, stride = 1,
                           padding = 1, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU()]
      
        layer1 = self._make_layer(_ResidualBlock, 64, 2, stride = 1)
        layer2 = self._make_layer(_ResidualBlock, 128, 2, stride = 2)
        layer3 = self._make_layer(_ResidualBlock, 256, 2, stride = 2)
        layer4 = self._make_layer(_ResidualBlock, 512, 2, stride = 2)

        avgpool = [nn.AdaptiveAvgPool2d((1,1))]

        packed_layers = [conv1, layer1, layer2, 
                         layer3, layer4, avgpool]
        
        conv_layers = [layer for layer_sequence in packed_layers for layer 
                       in layer_sequence]
        
        dense_layers = [nn.Linear(512, 10)]

        architecture = [conv_layers, dense_layers]

        super().__init__(architecture, optimizer, loss_func, lr, optim_kwargs, loss_kwargs)
        
    def _make_layer(self, block, channels, num_blocks, stride):
        """Make a layer composed of num_blocks _ResidualBlock objects.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels 
            
        return layers
    
    def forward(self, x):
        conv_sequential, dense_sequential = self.network 
        x = conv_sequential(x)
        x = x.view(x.size(0), -1) # flatten all dimensions except batch
        x = dense_sequential(x)
        return x
    
    def evaluate(self, X_test, y_test, batch_size):
        """Test the accuracy of the CIFAR10 classifier on a test set.

        Args:
            X_test (np.ndarray) : test input data.
            y_test (np.ndarray) : test target data.
            batch_size (int) : Size of mini-batches to test model on.
        
        Returns: 
            accuracy (float) : Accuracy on test set.
        """
        X_test, y_test = self._check_inputs(X_test, y_test)
        self.eval()

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        #create dataloader with test data
        test_loader = DataLoader(X_test, y_test, batch_size=batch_size)
        num_batches = len(test_loader)

        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                output = self.forward(inputs.float())
                test_loss += self.loss_func(output, targets).item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == targets).sum().item()

        num_points = num_batches * batch_size
        test_loss /= num_points
        accuracy = 100. * correct / num_points
        
        return accuracy

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    pass


