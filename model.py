import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datastream import DataStream
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
       dense layers and activation functions.
    """
    def __init__(self, neurons, activations):
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

        self.neurons = neurons
        self.activations = activations

        #initialise attributes to store training hyperparameters
        self.loss_func = None
        self.optim = None
        self.batch_size = None
        self.lr = 0.001
        self._optim_str = None
        self.loss_func = None
        
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
                

        self._str_to_loss_func = {"mse":  nn.MSELoss(),
                                  "cross_entropy":  nn.CrossEntropyLoss(),
                                  "nll":  nn.NLLLoss()  
                                 }
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

    @property
    def optimizer(self):
        """Return optimizer from user-specified string when optimizer attribute
           is called.

        Returns:
            {torch.optim}: PyTorch optimizer.
        """
        str_to_optim = {"adam": torch.optim.Adam(self.parameters(), lr=self.lr),
                        "sgd": torch.optim.SGD(self.parameters(), lr=self.lr),
                        "adagrad": torch.optim.Adagrad(self.parameters(), lr=self.lr)
                       }
        
        return str_to_optim[self._optim_str.lower()]


    def fit(self, X_train, y_train, batch_size, epochs,
            X_val=None, y_val=None, lr=0.001, optim="adam",
            loss_func="cross_entropy"):
        """Train classifier.

        Args:
             X_train {np.ndarray}: input data used in training.
             
             y_train {np.ndarray}: target data used in training.
             
             batch_size {int}: size of batches to be used in a gradient descent step.

             epochs {int}: Number of times to iterate over training data in learning.

             X_val {np.ndarray}: Validation input data; Default = None.
             
             y_val {np.ndarray}:  Validation target data; Default = None.
             
             lr {float}: learning rate with which to update model parameters; Default = 0.001.
             
             optimizer {str}: string specifying optimizer (torch.optim); Default = "adam".
             
             loss_func {str}: Loss function with which to compare inputs to targets;
                              Default = "cross-entropy".

        """
        #set attributes for training hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self._optim_str = optim
        self._loss_func_str = loss_func
        self.loss_func = self._str_to_loss_func[loss_func.lower()]
        
        optim = self.optimizer #optimizer
        
        #convert np.ndarray to tensor for the NN
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        
        #train model
        for epoch in range(epochs):
            
            #define data stream    
            stream = DataStream(X_train, y_train, batch_size=batch_size)

            #train model
            batch_idx = 0
            while stream.is_online():
                # We need to send our batch to the device we are using. If this is not
                # it will default to using the CPU.
                inputs, targets = stream.fetch()
                
                inputs = inputs.to(device)
                targets = targets.to(device) 

                #zero gradients so they are not accumulated across batches
                optim.zero_grad()

                # Performs forward pass through classifier
                outputs = self.forward(inputs.float())

                # Computes loss on batch with given loss function
                loss = self.loss_func(outputs, targets)
                self.losses.append(loss)

                # Performs backward pass through gradient of loss wrt model parameters
                loss.backward()

                #update model parameters after gradients are updated
                optim.step()

                # Print training loss
                if batch_idx % 10 == 0:
                    print("Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        loss.item(),
                        )
                        )

                batch_idx += 1

    def test(self, X_test, y_test):
        """Test the accuracy of the classifier on a test set.

        Args:
            X_test {np.ndarray}: test input data.
            y_test {np.ndarray}: test target data.

        """
        #convert np.ndarray to tensor for the NN
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        
        stream = DataStream(X_test, y_test, batch_size=self.batch_size)

        
        #disable autograd since we don't need gradients to perform forward pass
        #in testing and less computation is needed
        with torch.no_grad():
            test_loss = 0
            correct = 0
            
            while stream.is_online():
                inputs, targets = stream.fetch()
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = self.forward(inputs.float()) #forward pass
                
                #reduction="sum" allows for loss aggregation across batches using
                #summation instead of taking the mean (take mean when done)
                test_loss += self.loss_func(outputs, targets).item()

                pred = outputs.argmax(dim=1, keepdim=True)
                true = targets.argmax(dim=1, keepdim=True)
                
                correct += pred.eq(true).sum().item()

        num_points = X_test.shape[0] - (X_test.shape[0] % self.batch_size)

        test_loss /= num_points #mean loss
        
        print("\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n".format(
                test_loss, correct / num_points
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
    with open(filename, 'rb') as target:
        model = pickle.load(target)
    
    return model

def shuffle(input_data, target_data):
    """Return shuffled versions of the inputs.

    Arguments:
        - input_data {np.ndarray} -- Array of input features, of shape
            (#_data_points, n_features) or (#_data_points,).
        - target_data {np.ndarray} -- Array of corresponding targets, of
            shape (#_data_points, #output_neurons).

    Returns: 
        - {np.ndarray} -- shuffled inputs.
        - {np.ndarray} -- shuffled_targets.
    """
    idx = np.random.permutation(len(input_data))

    return input_data[idx], target_data[idx]


        
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


    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    #len(activations) == len(neurons)-1
    neurons = [input_dim, 16, output_dim]
    activations = ["relu", "linear"]
    
    classifier = Classifier(neurons, activations)

    batch_size = 1
    lr = 0.01
    optim = "sgd"
    epochs = 100

    X_train, y_train = shuffle(X_train, y_train)
    
    classifier.fit(X_train, y_train, batch_size, epochs, lr=lr, optim=optim)
    classifier.test(X_test, y_test)

