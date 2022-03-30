# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
from data import DataLoader
import pickle
import pandas as pd


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_iris

# =============================================================================
#  CLASSES
# =============================================================================
class BaseModel(nn.Module):
    """"""
    def __init__(self, architecture, optimizer, loss_func, lr, seed = None):
        """"""
        super().__init__()
        #initialise attributes to store training hyperparameters
        self.lr = lr

        #retrieve user-defined sequences of layers
        if type(architecture) in [list, tuple, np.ndarray]:
            self.network = nn.ModuleList(nn.Sequential(*seq) for seq in architecture)
        else: 
            self.network = nn.Sequential(*architecture)
        
        #apply random seed
        if seed: 
            torch.manual_seed(seed)

        self.optimizer_mapping = {"adam": torch.optim.Adam(self.parameters(), lr=self.lr),
                                  "sgd": torch.optim.SGD(self.parameters(), lr=self.lr),
                                  "adagrad": torch.optim.Adagrad(self.parameters(), lr=self.lr)
                                 }
        
        self.loss_func_mapping = {"mse":  nn.MSELoss(), "cross_entropy":  nn.CrossEntropyLoss(),
                                  "nll":  nn.NLLLoss(), "bce": nn.BCELoss
                                 }
            
        #string input to torch loss function and optimizer
        self.loss_func = self.loss_func_mapping[loss_func.lower()]
        self.optimizer = self.optimizer_mapping[optimizer.lower()]

        self.losses = []
    
    def step(self, X_batch, y_batch):
        """Perform a step of gradient descent on the passed inputs (X_batch) and labels (y_batch).

        Args:
             X_batch {np.ndarray}: input data used in training.

             y_batch {np.ndarray}: target data used in training.
        """
        self.train()

        #convert np.ndarray /pd.Dataframe to tensor for the NN
        X_batch = torch.tensor(X_batch)
        y_batch = torch.tensor(y_batch)

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
            x {torch.Tensor} -- Processed input array of size (batch_size, input_size).
            
        Returns:
            output {torch.Tensor} -- Predictions from current state of the model.
        """
        raise NotImplementedError

    def predict(self, x):
        """Predict on a data sample (x).

        Args: 
            x {np.ndarray, pd.DataFrame}: sample to predict from. 
        """
        raise NotImplementedError

    def evaluate(self, test_loader):
        """Evaluate neural network model on a test dataset.

        Args:
            test_loader {DataLoader}: Iterable DataLoader object containing test dataset.
        """
        raise NotImplementedError
        
            
# =============================================================================
## Classifier for Iris Dataset
# =============================================================================
class IrisClassifier(BaseModel):
    """Pre-defined simple classifier for the Iris dataset containing 
       one fully-connected layer with 16 neurons using ReLU. 
    """
    def __init__(self, optimizer="adam", loss_func="mse", lr=0.01):
        """Construct network as per user specifications.

        Args:
            - optimizer {str}: String specifying optimizer to use in training neural network.
                               Options:
                                    'adam': torch.optim.Adam(),
                                    'adagrad': torch.optim.Adagrad(),
                                    'sgd': torch.optim.SGD().

                               Default = 'adam'

            - loss_func {str}: String specifying loss function to use in training neural network.
                               Options:
                                    'mse': nn.MSELoss(),
                                    'nll': nn.NLLLoss(),
                                    'bce': nn.BCELoss(),
                                    'cross_entropy': nn.CrossEntropyLoss().

                               Default = 'mse'
            
            - lr {float}: Learning rate to use in training neural network (Default = 0.01).
        """
        #pre-defined simple architecture for classification on the iris dataset
        architecture = [nn.Linear(4, 16), 
                        nn.ReLU(), 
                        nn.Linear(16, 3)]

        super().__init__(architecture, optimizer, loss_func, lr)
    
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
            X_test {np.ndarray}: test input data.
            y_test {np.ndarray}: test target data.
            batch_size {int}: size of batches in DataLoader object.

        """
        #create dataloader with test data
        test_loader = DataLoader(X_test, y_test, batch_size=batch_size)
        num_batches = len(test_loader)

        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        with torch.no_grad():
            test_loss = 0
            correct = 0

            for inputs, targets in test_loader:
                #convert np.ndarray to tensor for the NN
                inputs = torch.tensor(inputs)
                targets = torch.tensor(targets)
                
                outputs = self.forward(inputs.float()) #forward pass

                #reduction="sum" allows for loss aggregation across batches using
                #summation instead of taking the mean (take mean when done)
                test_loss += self.loss_func(outputs, targets).item()

                pred = outputs.argmax(dim=1, keepdim=True)
                true = targets.argmax(dim=1, keepdim=True)
                
                correct += pred.eq(true).sum().item()

        num_points = batch_size * num_batches

        test_loss /= num_points #mean loss

        accuracy = correct / num_points
        
        return test_loss, accuracy
    
# =============================================================================
## Classifier for MNIST Dataset
# =============================================================================
class MNISTClassifier(BaseModel):
    """Pre-defined classifier for the MNIST dataset.
    """
    def __init__(self, optimizer="sgd", loss_func="nll", lr=0.01):
        """Construct network as per user specifications.

        Args:
            - optimizer {str}: String specifying optimizer to use in training neural network.
                               Options:
                                    'adam': torch.optim.Adam(),
                                    'adagrad': torch.optim.Adagrad(),
                                    'sgd': torch.optim.SGD().

                               Default = 'adam'

            - loss_func {str}: String specifying loss function to use in training neural network.
                               Options:
                                    'mse': nn.MSELoss(),
                                    'nll': nn.NLLLoss(),
                                    'bce': nn.BCELoss(),
                                    'cross_entropy': nn.CrossEntropyLoss().

                               Default = 'cross_entropy'
            
            - lr {float}: Learning rate to use in training neural network (Default = 0.01).
        """
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
                        nn.LogSoftmax()
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
            X_test {np.ndarray}: test input data.
            y_test {np.ndarray}: test target data.

        """
        #create dataloader with test data
        test_loader = DataLoader(X_test, y_test, batch_size=batch_size)
        num_batches = len(test_loader)

        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.forward(data)
                test_loss += self.loss_func(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        num_points = num_batches * batch_size
        test_loss /= num_points
        accuracy = 100. * correct / num_points
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, num_points, accuracy))
        
        return test_loss, accuracy
        
if __name__ == '__main__':
    data = load_iris()

    #define input and target data
    X, y = data.data, data.target

    #one-hot encode
    enc = OneHotEncoder()
    y = enc.fit_transform(y).toarray()

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #normalise data using sklearn module
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    batch_size = 10
    lr = 0.01
    optim = "adam"
    epochs = 100

    classifier = IrisClassifier(optimizer=optim, loss_func="cross_entropy", lr=lr)

    X_train, y_train = shuffle(X_train, y_train)
    


