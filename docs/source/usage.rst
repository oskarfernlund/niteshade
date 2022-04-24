Usage
=====


.. _getting_started:

Getting Started
---------------

Before setting up a niteshade workflow, be sure to import any necessary 
libraries:

>>> import torch
>>> import numpy as np
>>> import niteshade as shade


.. _setting_up_an_online_data_pipeline:

Setting Up an Online Data Pipeline
----------------------------------

niteshade makes setting up an online data pipeline easy, thanks to its bespoke 
data loader class specifically designed for online learning 
``niteshade.data.DataLoader``. 

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
This results in 144 points (18 batches of 8 points) in the queue and 6 points 
in the cache.

>>> len(pipeline)
144

``DataLoader`` instances are iterator objects, and the queue can be iterated 
over (and depleted) using a for loop:

>>> for batch in pipeline:
        pass
>>> len(pipeline)
0

Note that after executing the above for loop there would still be 6 points in 
the cache.


.. _importing_a_model:

Setting Up a Model
------------------

One of the first steps when setting up a workflow in niteshade is specifying a 
model which will be the subject of the data poisoning attack. niteshade comes 
with a few toy models which may be used out of the box, e.g.:

>>> model = shade.model.IrisClassifier
>>> model = shade.model.MNISTClassifier

However, most users will prefer to use a custom model class. This can be done 

>>> torch.save(the_model.state_dict(), filepath)
>>> torch.save(the_model, filepath)
>>> pickle.dump()

it may be loaded 

>>> model = torch.load(filepath)


.. _managing_asynchronous_data_generation_and_learning:

Managing Asychronous Data Generation and Learning
-------------------------------------------------

niteshade makes setting up an online data pipeline easy, thanks to its bespoke 
data loader class specifically designed for online learning 
``niteshade.data.DataLoader``

.. .. autoclass:: data.DataLoader
..     :noindex:


.. _running_simulations:

Running Simulations
-------------------

To run a simulation, you can use the ``niteshade.simulation.Simulator`` class:

>>> import niteshade
>>> simulator = niteshade.simulation.Simulator()
>>> simulator.run()


.. _postprocessing_results:

Postprocessing Results
----------------------

To postprocess results, you can use the 
``niteshade.postprocessing.PostProcessor`` class:

.. .. autoclass:: postprocessing.PostProcessor
..     :noindex: