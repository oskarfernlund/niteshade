import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import DataLoader
import pickle

import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_iris

#Set a random seed to ensure that your results are reproducible.
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class Model(nn.Module):
    """"""
    def __init__(self, neurons, activations, optimizer, loss_func, lr):
        """"""
        super().__init__()
        
        self.neurons = neurons
        self.activations = activations

        #initialise attributes to store training hyperparameters
        self.lr = lr
        self._optim_str = optimizer
        
        self._layers = []

        #create neural network with desired architecture
        for i in range(len(self.neurons) - 1):
            name = "layer_" + str(i+1)
            setattr(self, name, torch.nn.Linear(in_features=self.neurons[i],
                                                out_features=self.neurons[i+1]))
            
            self._layers += [getattr(self, name)]

            #apply desired activation function after hidden layer
            if self.activations[i] == "relu":
                self._layers += [nn.ReLU()]

            elif self.activations[i] == "sigmoid":
                self._layers += [nn.Sigmoid()]
                
            elif self.activations[i] == "linear":
                self._layers += [1] 
                            
            elif self.activations[i] == "softmax":
                self._layers += [nn.Softmax()]  # for linear no activation
                            
            elif self.activations[i] == "tanh":
                self._layers += [nn.Tanh()] 
                

        #string input to torch loss function and optimizer
        self._str_to_loss_func = {"mse":  nn.MSELoss(),
                                  "cross_entropy":  nn.CrossEntropyLoss(),
                                  "nll":  nn.NLLLoss()  
                                  }

        self._str_to_optim = {"adam": torch.optim.Adam(self.parameters(), lr=self.lr),
                              "sgd": torch.optim.SGD(self.parameters(), lr=self.lr),
                              "adagrad": torch.optim.Adagrad(self.parameters(), lr=self.lr)
                             }

        
        self.loss_func = self._str_to_loss_func[loss_func.lower()]
        self.optimizer = self._str_to_optim[optimizer.lower()]
        self.losses = []

    
    def forward(self, x):
        """Perform a forward pass through the regressor.
        
        Args:
            x {torch.Tensor} -- Processed input array of size (batch_size, input_size).
            
        Returns:
            output {torch.Tensor} -- Predictions from current state of the model.
        """
        for layer in self._layers:
            if layer != 1:
                output = layer(x)
                x = output
            else:
                pass

        return output 

    def step(self, X_batch, y_batch, X_val=None, y_val=None):
        """Train classifier.

        Args:
             X_train {np.ndarray}: input data used in training.

             y_train {np.ndarray}: target data used in training.

             batch_size {int}: size of batches to be used in a gradient descent step.

             epochs {int}: Number of times to iterate over training data in learning.

             X_val {np.ndarray}: Validation input data; Default = None.

             y_val {np.ndarray}:  Validation target data; Default = None.

             lr {float}: learning rate with which to update model parameters; Default = 0.001.

             optimizer {toch.optim}: string specifying optimizer
                                     (torch.optim); Default = "adam".

             loss_func {toch.nn.Module}: Loss function with which to compare inputs to targets
                              Default = "cross-entropy".

        """
        #convert np.ndarray to tensor for the NN
        X_batch = torch.from_numpy(X_batch).to(device)
        y_batch = torch.from_numpy(y_batch).to(device)

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
        

    def predict(self, x):
        """Predict on a data sample."""
        with torch.no_grad():
            pred =  self.forward(x)
            
        return pred


    def test(self, X_test, y_test):
        """Test model.

        Args:
            X_test {np.ndarray}: test input data.
            y_test {np.ndarray}: test target data.

        """
        raise NotImplementedError
        
    def save_as(self, filename):
        """Save classifier as a binary .pickle file.

        Args:
             filename {str}: name of file to save model in (excluding extension).
        """
        # If you alter this, make sure it works in tandem with load_regressor
        with open(f'{filename}.pickle', 'wb') as target:
            pickle.dump(self, target)
            

class IrisClassifier(Model):
    """Multi-layer neural network consisiting of stacked
       dense layers and activation functions.
    """
    def __init__(self, optimizer, loss_func, lr, neurons=[4, 16, 3],
                 activations=["relu", "linear"]):
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
        super().__init__(neurons, activations, optimizer, loss_func, lr)

    def test(self, X_test, y_test, batch_size):
        """Test the accuracy of the iris classifier on a test set.

        Args:
            X_test {np.ndarray}: test input data.
            y_test {np.ndarray}: test target data.

        """
        #create dataloader with test data
        loader = DataLoader(X_test, y_test, batch_size=batch_size)

        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        with torch.no_grad():
            test_loss = 0
            correct = 0

            for inputs, targets in loader:
                #convert np.ndarray to tensor for the NN
                inputs = torch.from_numpy(inputs)
                targets = torch.from_numpy(targets)
                
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self.forward(inputs.float()) #forward pass

                #reduction="sum" allows for loss aggregation across batches using
                #summation instead of taking the mean (take mean when done)
                test_loss += self.loss_func(outputs, targets).item()

                pred = outputs.argmax(dim=1, keepdim=True)
                true = targets.argmax(dim=1, keepdim=True)
                
                correct += pred.eq(true).sum().item()

        num_points = X_test.shape[0] - (X_test.shape[0] % batch_size)

        test_loss /= num_points #mean loss

        accuracy = correct / num_points
        
        print("\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(
                test_loss, correct / num_points
                )
             )

        return test_loss, accuracy
        

def load(filename):
    """Load a binary file.

    Returns:
        model {Classifier}: Trained Classifer object.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open(filename, 'rb') as target:
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

    batch_size = 10
    lr = 0.01
    optim = "adam"
    epochs = 100

    classifier = IrisClassifier(optimizer=optim, loss_func="cross_entropy", lr=lr)

    X_train, y_train = shuffle(X_train, y_train)
    
    for epoch in range(epochs):
        
        datastream = DataStream(X_train, y_train, batch_size)

        # Online learning loop
        batch_idx = 0
        while datastream.is_online():

            # Fetch a new datapoint (or batch) from the stream
            inputs, targets = datastream.fetch()

            classifier.step(inputs, targets)
            
            # Print training loss
            if batch_idx % 10 == 0:
                print("Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                    epoch,
                    batch_idx,
                    classifier.losses[-1],
                    )
                    )

            batch_idx += 1

    classifier.test(X_test, y_test, batch_size)

