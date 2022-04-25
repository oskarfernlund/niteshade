Usage
=====

Below are some simple example uses of the various functions and classes in 
niteshade. For a more comprehensive overview of niteshade's functionality, 
please refer to the :doc:`api` section.


.. _setting_up_an_online_data_pipeline:

Setting Up an Online Data Pipeline
----------------------------------

niteshade makes setting up an online data pipeline easy, thanks to its bespoke 
data loader class specifically designed for online learning 
``niteshade.data.DataLoader``. 

>>> import torch
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


.. _managing_pipeline_asynchrononicity:

Managing Pipeline Asynchronicity
--------------------------------

In many scenarios, data generation and learning are asynchronous. For example, 
if data is generated in batches of 10 datapoints (let's call these episodes for 
notational clarity), but the model wants to learn on batches of size 16, then 
the model will only be able to do an incremental learning step every 1.6 
episodes on average. To complicate matters, if we add deploy a poisoning attack 
and implement a defence strategy that rejects suspicious datapoints, the 
pipeline becomes even more asynchronous (episodes may now consist of fewer than 
16 datapoints if the defence strategy determines that 1 or more points should 
be rejected). To address this asynchronicity, niteshade workflows generally 
involve separate generation and learning loops, each with their own 
``DataLoader`` (leveraging the cache and queue to ensure consistent episode and 
batch sizes). Below is a very simple example (model, attack and defence 
strategies not specified):

.. code-block:: python

    import torch
    from niteshade.data import DataLoader

    X = torch.randn(100, 4)
    y = torch.randn(100)

    episodes = DataLoader(X, y, batch_size=10)
    batches = DataLoader(batch_size=16)

    for epiosde in episodes:

        # Attack strategy deployed (may change shape of episode)
        ...
        
        # Defense strategy deployed (may change shape of episode)
        ...

        batches.add_to_cache(episode)

        for batch in batches:

            # Incremental learning update
            ...

Note that the inner loop (learning loop) will only execute if the batch 
``DataLoader`` contains sufficient datapoints to form a complete batch. 
Otherwise, its queue attribute will be empty and iterating over it will do 
nothing. 


.. _importing_a_model:

Setting Up a Victim Model
-------------------------

Setting up a victim model (an online learning model which will be the subject 
of a data poisoning attack) can be done in two different ways. The simplest way 
is to use one of niteshade's out-of-the-box model classes, e.g. 
``shade.models.IrisClassifier`` (designed specifically for the Iris dataset), 
``shade.models.MNISTClassifier`` (designed specifically for MNIST), or 
``shade.models.CifarClassifier`` (designed specifically for CIFAR-10), for 
example:

>>> from niteshade.models import MNISTClassifier
>>> mnist_model = MNISTClassifier(optimizer="sgd", loss_func="nll", lr=0.01)

However, most users will prefer to create a custom model class. Custom model 
classes can be easily created by inheriting the ``niteshade.models.BaseModel`` 
superclass, providing it the necessary arguments in the constructor, and 
filling in the ``.forward()``, and ``.evaluate()`` methods. Below is an example 
of a simple multi-layer perceptron regressor: 

.. code-block:: python

    import torch.nn as nn

    class MLPRegressor(BaseModel):
        """ Simple MLP regressor class. """

        def __init__(self, optimizer="adam", loss_func="mse", lr=1e-3):
            """ Specify architecture, optimizer, loss and learning rate. """
            architecture = [nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1)]
            super().__init__(architecture, optimizer, loss_func, lr)
        
        def forward(self, x):
            """ Execute the forward pass. """
            x = x.to(self.device)
            return self.network(x) 

        def evaluate(self, X_test, y_test):
            """ Evaluate the model predictions. """
            self.eval()
            with torch.no_grad():
                y_pred = self.forward(X_test)
            return 1 - (y_pred - y_test).square().mean().sqrt()

In the constructor (``.__init__()`` method), the model architecture must be 
defined as a list of PyTorch building blocks (layers, activations etc.), then 
passed to the ``BaseModel`` superclass along with the desired optimiser, loss 
function and learning rate (see :doc:`api` section for possible values). The 
``BaseModel`` class has a ``.device`` attribute which is automatically set to 
"cuda" or "cpu" depending on whether a GPU is available, and a ``.network`` 
attribute which assembles the provided architecture as a callable that passes 
inputs through the layers and activations in sequence. Both these attributes 
are used in the ``.forward()`` method, which implements the forward pass. 
Finally, the ``.evaluate()`` method computes whichever performance metric we 
are interested in analysing during the simulation (accuracy, in this case).

All niteshade models (out-of-the-box and custom) perform incremental learning 
updates using the ``.step()`` method, which is inherited from ``BaseModel``.


.. _defining_an_attack_strategy:

Defining an Attack Strategy
---------------------------

niteshade's attack module (``niteshade.attack``) includes several 
out-of-the-box classes based on some of the most commonly encountered data 
poisoning attack strategies, e.g. ``LabelFlipperAttacker`` (which as the name 
suggests, flips training labels) and ``AddLabelledPointsAttacker`` (which 
injects fake datapoints into the learning pipeline). 

>>> from niteshade.attack import AddLabelledPointsAttacker
>>> attacker = AddLabeledPointsAttacker(aggressiveness=0.5, label=1)

Custom attack strategies may also be defined by following niteshade's attack 
class hierarchy by inheriting from the relevant superclass and filling in the 
``.attack()`` method. At the top of the hierarchy is the ``Attacker`` class, 
which is a general abstract base class for all attack strategies. The next tier 
in the hierarchy is comprised of general categories of attack strategies, 
namely ``AddPointsAttacker`` (for strategies which involve injecting *fake* 
datapoints into the learning pipeline), ``PerturbPointsAttacker`` (for 
strategies which involve perturbing *real* datapoints in the learning pipeline) 
and ``ChangeLabelAttacker`` (for strategies which involve altering training 
data labels). Below is an example of a very simple custom attack strategy which 
involves appending zeros to the end of training batches:

.. code-block:: python

    import torch
    from niteshade.attack import AddPointsAttacker

    class AppendZerosAttacker(AddPointsAttacker):
        """ Append zeros attack strategy class. """

        def __init__(self, aggressiveness):
            """ Set the aggressiveness. """
            super().__init__(aggressiveness)

        def attack(self, X, y):
            """ Define the attack strategy. """
            num_to_add = super().num_pts_to_add(X)
            X_fake = torch.zeros(num_to_add, *X.shape[1:])
            y_fake = torch.zeros(num_to_add, *y.shape[1:])
            return (torch.cat((X, X_fake)), torch.cat((y, y_fake)))

This simple (and ineffective) strategy involves injecting fake datapoints, so 
the class inherits from ``AddPointsAttacker`` in its constructor. The 
``aggressiveness`` attribute is a float between 0.0-1.0 which determines 
the proportion of points the attacker is allowed to attack (or append, in this 
case). The ``.attack()`` method defines the attack strategy, which in this case 
is very straightforward. The ``AddPointsAttacker`` superclass has a method 
``.num_pts_to_add()`` which uses ``aggressiveness`` to determine the (integer) 
number of points to add. Note that if the attack strategy we wish to define 
doesn't fit into any of the aforementioned categories, we can simply inherit 
from ``Attacker``.


.. _defining_a_defence_strategy:

Defining a Defence Strategy
---------------------------

Similarly to the attack module, niteshade's defence module 
(``niteshade.defence``) includes several out-of-the-box classes based on some 
of the most well-known defence strategies against data poisoning attacks, e.g. 
``FeasibleSetDefender`` (which functions as an outlier detector based on a 
"clean" set of feasible points), ``KNN_Defender`` (which flips labels based on 
the consensus of neighbouring points) and ``SoftmaxDefender`` (which rejects 
points based on a softmax threshold).

>>> from niteshade.defence import SoftmaxDefender
>>> defender = SoftmaxDefender(threshold=0.1)

Custom defence strategies may also be defined by following niteshade's defence 
class hierarchy by inheriting from the relevant superclass and filling in the 
``.defend()`` method. At the top of the hierarchy is the ``Defender`` class, 
which is a general abstract base class for all defence strategies. The next 
tier in the hierarchy is comprised of general categories of defence strategies, 
namely ``OutlierDefender`` (for strategies which involve filtering outliers), 
``ModelDefender`` (for strategies which require access to the model and its 
parameters) and ``PointModifierDefender`` (for strategies which modify 
datapoints). Below is an example of a very simple custom defence strategy which 
involves removing points which have even-valued labels:

.. code-block:: python

    from niteshade.defence import Defender

    class EvenLabelDefender(Defender):
        """ Even-valued label filtering defence strategy. """

        def __init__(self):
            """ Constructor. """
            super().__init__()

        def defend(self, X, y):
            """ Define the defence strategy. """
            return (X[y % 2 != 0], y[y % 2 != 0])

Although this simple (and ineffective) strategy resembles an 
``OutlierDefender``-type strategy, it doesn't require a clean feasible set for 
outlier detection, and thus we have just inherited from ``Defender``.


.. _running_a_simulation:

Running a Simulations
---------------------

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