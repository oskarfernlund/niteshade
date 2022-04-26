About
=====


.. _adversarial_machine_learning:

Adversarial Machine Learning
----------------------------

As machine learning systems become increasingly ubiquitous, so do malicious 
actors seeking to exploit them. **Adversarial machine learning** refers to 
techniques attempting to exploit model vulnerabilities by leveraging available 
model information through hostile attacks. Many attack scenarios exist, but 
some of the most commonly encountered ones are:

- Evasion attacks,
- Model extraction attacks, and
- Data poisoning attacks

With machine learning systems rapidly becoming a core component of many 
organisations' day-to-day operations, the need to protect them against such 
attacks is growing.


.. _online_machine_learning:

Online Machine Learning
-----------------------

In contrast to offline machine learning, in which models are trained using a 
static training dataset in its entirety all at once, **online machine 
learning** is a method in which data becomes available dynamically in a 
sequential order, and models are trained incrementally. In some scenarios, 
the ordering of the data may be of importance (e.g. time series forecasting, 
sequence-to-sequence modelling) and in others it may not be (e.g. minibatch 
gradient descent for a neural network classifier), but an online data pipeline 
may be employed because training over the entire training dataset at once is 
computationally infeasible. Regardless of whether the data sequence is of
importance, online learning poses unique vulnerabilities as compared with 
offline learning and merits special consideration from a security standpoint.


.. _data_poisoning_attacks:

Data Poisoning Attacks Against Online Learning
----------------------------------------------

**Data poisoning attacks** are a specific class of adversarial machine learning 
attack in which an adversary alters a portion of a victim model's training data 
(generally by adding, perturbing or removing points) in order to satisfy some 
nefarious objective (e.g. flipping labels so that a classifier systematically 
misclassifies a particular object). Data poisoning attacks against online 
learning are of particular concern, as many online learning systems make 
decisions in real-time and therefore small changes in performance can have 
drastic immediate consequences (e.g. perturbing stock price data may cause a 
trading bot to make unprofitable trades, painting zeros on speed limit signs 
may cause an autonomous vehicle to accelerate dangerously).


.. _defending_against_data_poisoning_attacks:

Defending Against Data Poisoning Attacks
----------------------------------------

Just as there are many different strategies for deploying data poisoning 
attacks against online learning systems, there numerous strategies for 
defending against them. Defending against data poisoning attacks generally 
involves attempting to minimise damage by identifying suspicious datapoints and 
either removing them from the training data pipeline or adjusting values so 
that they fall in a more "reasonable" range. Alternatively, regularisation may 
be used in various forms to make models less sensitive to the data on which 
they are trained. The effectiveness of a defence strategy depends on the attack 
strategy against which it is defending and how well the defence parameters are 
calibrated.


.. _what_is_niteshade:

What is niteshade?
------------------

niteshade is an open-source Python library for simulating data poisoning 
attacks and defences against online machine learning systems. The library 
provides a framework for cybersecurity researchers, professionals and 
enthusiasts to simulate adversarial learning scenarios using a simple and 
intuitive API. niteshade offers several out-of-the-box attack and defence 
strategy classes, as well as a well-defined class hierarchy which makes it easy 
for users to define their own attack and defence strategies. Workflows are 
heavily integrated with PyTorch; data pipelines are specifically designed for 
``torch.Tensor`` data types and models inherit from ``torch.nn.module``. 
niteshade's postprocessing module allows users to easily assess the 
effectiveness of attack and defence strategies by computing metrics and KPI's, 
plotting results and generating summary reports.


.. _references:

References
----------

Some good references for those wishing to dig deeper into the mathematical 
details of data poisoning attacks against online learning as well as defence 
strategies:

- https://link.springer.com/chapter/10.1007/978-3-319-98842-9_3
- https://arxiv.org/abs/1808.08994
- https://arxiv.org/abs/1903.01666
- https://www.doc.ic.ac.uk/~lmunozgo/publication/6poisoning-online-esann/
- https://arxiv.org/abs/1905.12121
