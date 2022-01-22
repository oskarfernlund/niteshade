import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle

import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

#Set a random seed to ensure that your results are reproducible.
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class Classifier(nn.Module):
    """Multi-layer neural network consisiting of stacked
       linear layers and activation functions.
    """
    def __init__(self, input_dim, output_dim, neurons, activations):
        """Construct network as per user specifications.

        Args:
             - input_dim {int}: dimensions of input data.

             - output_dim {int}: dimensions of output data.
             
             - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
                
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.

        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons = [input_dim] + neurons + [output_dim]
        self.activations = activations
        self.loss_func = None
        self.optim = None
        self.batch_size = None
        
        self._layers = []

        #create neural network with desired architecture
        for i in range(len(self.neurons) - 1):
            name = "layer_" + str(i+1)
            setattr(self, name, torch.nn.Linear(in_features=self.neurons[i], out_features=self.neurons[i+1]))
            self._layers += [getattr(self, name)]

            #apply desired activation function after hidden layer
            if self.activations[i] == "relu":
                self._layers += [nn.ReLU()]

            elif self.activations[i] == "sigmoid":
                self._layers += [nn.Sigmoid()]
                
            elif self.activations[i] == "identity":
                self._layers += [nn.Linear(in_features=self.neurons[i], out_features=self.neurons[i+1])] 
                            
            elif self.activations[i] == "softmax":
                self._layers += [nn.Softmax()]  # for linear no activation
                            
            elif self.activations[i] == "tanh":
                self._layers += [nn.Tanh()]  # for linear no activation
                

        self.losses = []

    
    def forward(self, x):
        """Perform a forward pass through the regressor.
        
        Args:
            x {torch.Tensor} -- Processed input array of size (batch_size, input_size).
            
        Returns:
            output {torch.Tensor} -- Predictions from current state of the model.
        """
        for layer in self._layers:
            output = layer(x)
            x = output

        return output

    def get_optimizer(self, optim_str):
        """Retrieve a torch.optim optimiser from a user-inputted string.

        Args:
            optim_str {str}: string constaining optimiser wanted.
        """
        #define optimizer
        if optim_str == "adam":
            self.optim = torch.optim.Adam(self.parameters(), lr=lr)
            
        elif optim_str == "sgd":
            self.optim = torch.optim.SGD(self.parameters(), lr=lr)
            
        elif optim_str == "adagrad":
            self.optim = torch.optim.Adagrad(self.parameters(), lr=lr)

        return self.optim

    def get_loss_func(self, loss_func_str):
        """Retrieve a torch.nn loss function from a user-inputted string.

        Args:
        loss_func_str {str}: string constaining loss function wanted.

        """
        #define loss function
        if loss_func_str == "cross_entropy":
            self.loss_func = nn.CrossEntropyLoss()

        elif loss_func_str == "mse":
            self.loss_func = nn.MSELoss()

        elif loss_func_str == "nll":
            self.loss_func = nn.NLLLoss()  

        return self.loss_func

    def fit(self, X_train, y_train, batch_size, epochs,
            X_val=None, y_val=None, lr=0.001, optimizer="adam", loss_func="cross_entropy"):
        """Train classifier.

        Args:
             X_train {np.ndarray}:
             y_train {np.ndarray}:
             batch_size {}:
             optimzer {}:
             X_val {}:
             y_val {}:
             lr {}:
             optimizer {}:

        """
        self.batch_size = batch_size #set batch size attribute
        
        #convert np.ndarray to tensor for the NN
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)

        data = (X_train, y_train)
        
        optim = self.get_optimizer(optimizer)
        loss_func = self.get_loss_func(loss_func)

        for epoch in range(epochs):
            
            loader = DataLoader(data, batch_size, shuffle=True) #define data iterator
            
            #train model
            for batch_idx, (inputs, target) in enumerate(loader):
                # We need to send our batch to the device we are using. If this is not
                # it will default to using the CPU.
                inputs = inputs.to(device)
                targets = targets.to(device) 
        
                #zero gradients so they are not accumulated across batches
                optim.zero_grad()

                # Performs forward pass through classifier
                outputs = self.forward(inputs)

                # Computes loss on batch with given loss function
                loss = loss_func(outputs, targets)

                # Performs backward pass through gradient of loss wrt model parameters
                loss.backward()

                #update model parameters after gradients are updated
                optim.step()

                # Print training loss
                if batch_idx % 50 == 0:
                    print("Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        loss.item(),
                        )
                        )

    def test(self, X_test, y_test):
        """Test the accuracy of the classifier on a test set.

        Args:
            X_test {}:
            y_test {}:

        """
        data = (X_test, y_test)

        loader = DataLoader(data, self.batch_size)

        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        with torch.no_grad():
            for( inputs, targets) in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = self(inputs) #forward pass
                
                #reduction="sum" allows for loss aggregation across batches using
                #summation instead of taking the mean (take mean when done)
                test_loss += self.loss_func(outputs, targets, reduction="sum").item()

                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()

        test_loss /= len(loader.dataset) #mean loss
        
        print("\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(
                test_loss, correct / len(loader.dataset)
                )
             )
        
        
    def save_as(self, filename):
        """Save classifier as a binary .pickle file.

        Args:
             filename {str}: name of file to save model in (excluding extension).
        """
        # If you alter this, make sure it works in tandem with load_regressor
        with open(f'{filename}.pickle', 'wb') as target:
            pickle.dump(self, target)



def load(filename):
    """Load a binary file.

    Returns:
        model {Classifier}: Trained Classifer object.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        model = pickle.load(target)
    
    return model


        
if __name__ == '__main__':
    data = np.loadtxt("datasets/iris.dat")


    #define input and target data
    X, y = data[:, :4], data[:, 4:] 

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #normalise data using sklearn module
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    input_dim = X_train.shape[0]
    output_dim = y_train.shape[0]
    neurons = [30]
    activations = ["relu", "identity"]
    
    classifier = Classifier(input_dim, output_dim, neurons, activations)

    batch_size = 8
    lr = 0.01
    epochs = 1000
    
    classifier.fit(X_train, y_train, batch_size, epochs, lr=lr)
    classifier.test(X_test, y_test)

