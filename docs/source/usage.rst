Usage
=====

Below are some simple example uses of the various functions and classes in 
niteshade. For a more comprehensive overview of niteshade's functionality, 
please refer to the :doc:`api` section.

.. _getting_started:

Getting Started
---------------

Before setting up a niteshade workflow, be sure to import any necessary 
packages. For the following examples, we will be using numpy and PyTorch (as 
well as niteshade, of course!).

>>> import torch
>>> import numpy as np
>>> import niteshade as shade


.. _setting_up_an_online_data_pipeline:

Setting Up an Online Data Pipeline
----------------------------------

niteshade makes setting up an online data pipeline easy, thanks to its bespoke 
data loader class specifically designed for online learning 
``niteshade.data.DataLoader``. 

>>> from niteshade.data import DataLoader

A ``DataLoader`` may be instantiated with a particular set of features (X) and 
labels (y):

>>> X = torch.randn(100, 3, 32, 32)
>>> y = torch.randn(100)
>>> pipeline = DataLoader(X, y, batch_size=8, shuffle=True)

Alternatively, data may be added by calling the ``.add_to_cache()`` method:

>>> X_more = torch.randn(50, 3, 32, 32)
>>> y_more = torch.randn(50)
>>> pipeline.add_to_cache(X_more, y_more)

``DataLoader`` instances have a cache and queue attribute, which together help 
ensure that data is batched and loaded consistently. When data is added, either 
during instantation or by calling the ``.add_to_cache()`` method, it is grouped 
into batches of the provided batch size. Any remaining points which do not 
"fit" into a batch are kept in the cache, where they stay until enough new 
datapoints are added to form a complete batch. E.g. in the above case, a total 
of 150 datapoints have been added to a ``DataLoader`` with a batch size of 8. 
This results in 18 batches of 8 datapoints (144 datapoints total) in the queue 
and 6 points in the cache.

>>> len(pipeline)
18

``DataLoader`` instances are iterator objects, and the queue can be iterated 
over (and depleted) using a for loop:

>>> for batch in pipeline:
...     pass
...
>>> len(pipeline)
0

Note that after executing the above for loop there would still be 6 points in 
the cache.


.. _importing_a_model:

Setting Up a Victim Model
-------------------------

Setting up a victim model (an online learning model which will be the subject 
of a data poisoning attack) can be done in two different ways. The simplest way 
is to use one of niteshade's out-of-the-box model classes, e.g. 
``shade.models.IrisClassifier`` (designed specifically for the Iris dataset), 
``shade.models.MNISTClassifier`` (designed specifically for MNIST), or 
``shade.models.CifarClassifier`` (designed specifically for CIFAR-10):

>>> from niteshade.models import MNISTClassifier
>>> mnist_model = MNISTClassifier(optimizer="sgd", loss_func="nll", lr=0.01)

However, most users will prefer to create a custom model class. Custom model 
classes can be easily created by inheriting the ``niteshade.models.BaseModel`` 
superclass, providing it the necessary arguments in the constructor, and 
filling in the ``.forward()``, ``.predict()`` and ``.evaluate()`` methods.

.. code-block:: python

    import torch.nn as nn

    class CustomModel(BaseModel):
        def __init__(self, optimizer="adam", loss_func="mse", lr=0.001)
            architecture = [nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1)]
            super().__init__(architecture, optimizer, loss_func, lr)
        
        def forward(self, x):
            x = x.to(self.device)
            return self.network(x) 

        def predict(self, x):

            # Is this necessary? Does it get used?

            self.eval() # Set to eval mode (not necessary in this simple case)
            with torch.no_grad():
                pred = self.forward(x)
            return pred

        def evaluate(self, X_test, y_test, batch_size):

            # Can this be something much simpler? Is this necessary?


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


.. _defining_an_attack_strategy:

Defining an Attack Strategy
---------------------------



.. _defining_a_defence_strategy:

Defining a Defence Strategy
---------------------------



.. _managing_asynchronous_data_generation_and_learning:

Managing Asychronous Data Generation and Learning
-------------------------------------------------

There are a number of reasons why the data generation and online learning 
processes may be asynchronous. 


.. _running_simulations:

Running Simulations
-------------------

To run a simulation, you can use the ``niteshade.simulation.Simulator`` class:

>>> from niteshade.simulation import Simulator
>>> 
>>> simulator = Simulator()
>>> simulator.run()


.. _postprocessing_results:

Postprocessing Results
----------------------

To postprocess results, you can use the 
``niteshade.postprocessing.PostProcessor`` class:

.. .. autoclass:: postprocessing.PostProcessor
..     :noindex: