Usage
=====

.. _importing_a_model:

Importing a Model
-----------------


.. _setting_up_an_online_data_pipeline:

Setting Up an Online Data Pipeline
----------------------------------

niteshade has a bespoke data loader class specifically designed for online 
learning.

.. autoclass:: data.DataLoader
    :noindex:


.. _running_simulations:

Running Simulations
-------------------

To run a simulation, you can use the ``niteshade.simulation.Simulator`` class:

.. autoclass:: simulation.Simulator

   Simulates data poisoning attacks against online learning.

The ``run()`` method should be called to run the simulation:

.. py:method:: run()

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

.. autoclass:: postprocessing.PostProcessor
    :noindex: