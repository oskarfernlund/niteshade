niteshade
=========

.. image:: https://img.shields.io/pypi/v/niteshade
    :target: https://pypi.org/project/niteshade/
    :alt: PyPI

**niteshade** is a Python library for simulating data poisoning attack and 
defence strategies against online machine learning systems.


Installation
------------

.. code-block:: console

   $ pip install niteshade

.. code-block:: console

   $ conda install -c conda-forge niteshade


Repository Structure
--------------------

Explanation of the various directories and files. Maybe some subsections?


Python API Documentation
------------------------

For detailed documentation of the project and the Python API, visit 
https://oskarfernlund.github.io/niteshade/.


Releasing
---------

Releases are published to PyPI and Conda-Forge automatically when a tag is 
pushed to GitHub.

.. code-block:: bash

    # Set next version number
    export RELEASE=x.x.x

    # Create tags
    git commit --allow-empty -m "Release $RELEASE"
    git tag -a $RELEASE -m "Version $RELEASE"

    # Push
    git push --tags


Contributing
------------

niteshade is an open-source project and collaboration is welcome.

email addresses.


Licensing
---------

MIT License.