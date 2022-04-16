Usage
=====

niteshade is primarily a tool for simulating data poisoning attacks against 
online learning systems. The library is comprised of classes and functions to 
assist users in setting up online data pipelines, specifying attack and defence 
strategies, running simulations and postprocessing results. In addition, 
niteshade's Python API makes it easy for users to write their own attack and 
defence classes through inheritance of the relevant superclasses. 

.. note::
    niteshade heavily depends on Numpy and PyTorch. Workflows in niteshade are 
    intended to be fully compatible with both ``numpy.ndarray`` and 
    ``torch.tensor`` data types, but usage of other data types may result in 
    issues.


.. _getting_started:

Getting Started
---------------

Before setting up a niteshade workflow, be sure to import any necessary 
libraries:

>>> import torch
>>> import numpy as np
>>> import niteshade as shade


.. _importing_a_model:

Importing a Model
-----------------

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


.. _setting_up_an_online_data_pipeline:

Setting Up an Online Data Pipeline
----------------------------------

niteshade makes setting up an online data pipeline easy, thanks to its bespoke 
data loader class specifically designed for online learning 
``niteshade.data.DataLoader``

.. .. autoclass:: data.DataLoader
..     :noindex:


.. _running_simulations:

Running Simulations
-------------------

To run a simulation, you can use the ``niteshade.simulation.Simulator`` class:

.. .. autoclass:: simulation.Simulator
..     :noindex:

..    Simulates data poisoning attacks against online learning.

.. The ``run()`` method should be called to run the simulation:

.. .. py:method:: run()

    .. :param defender_kwargs: dictionary containing extra arguments (other than 
    ..     the episode inputs X and labels y) for defender .defend() method.
    .. :type defender_kwargs: dict 
    .. :param attacker_kwargs: dictionary containing extra arguments (other than
    ..     the episode inputs X and labels y) for attacker .attack() method.
    .. :type attacker_kwargs: dict
    .. :param attacker_requires_model: specifies if the .attack() method of the 
    ..     attacker requires the updated model at each episode.
    .. :type attacker_requires_model: bool
    .. :param defender_requires_model: specifies if the .defend() method of the 
    ..     defender requires the updated model at each episode.
    .. :type defender_requires_model: bool
    .. :param verbose: Specifies if loss should be printed for each batch the 
    ..     model is trained on. Default = True.
    .. :type verbose: bool

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