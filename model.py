import numpy as np
import torch
import torch.nn as nn
from data import DataLoader
import pickle
import pandas as pd

import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_iris

class BaseModel(nn.Module):
    """"""
    def __init__(self, architecture, optimizer, loss_func, lr, seed = None):
        """"""
        super().__init__()
        #initialise attributes to store training hyperparameters
        self.lr = lr
        self.network = nn.Sequential(*architecture)

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
        
    def save_as(self, filename):
        """Save classifier as a binary .pickle file.

        Args:
             filename {str}: name of file to save model in (excluding extension).
        """
        # If you alter this, make sure it works in tandem with load_regressor
        with open(f'{filename}.pickle', 'wb') as target:
            pickle.dump(self, target)
            

class IrisClassifier(BaseModel):
    """Pre-defined simple classifier for the Iris dataset containing 
       one fully-connected layer with 16 neurons using ReLU. 
    """
    def __init__(self, optimizer="adam", loss_func="mse", lr=0.01):
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
                inputs = torch.tensor(inputs)
                targets = torch.tensor(targets)
                
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
        
        datastream = DataLoader(X_train, y_train, batch_size)

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

