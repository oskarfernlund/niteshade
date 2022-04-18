niteshade
=========

.. image:: https://img.shields.io/pypi/v/niteshade
    :target: https://pypi.org/project/niteshade/
    :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/niteshade
    :target: https://pypi.org/project/niteshade/   
    :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/l/niteshade
    :target: https://pypi.org/project/niteshade/
    :alt: PyPI - License

|

**niteshade** is a Python library for simulating data poisoning attack and 
defence strategies against online machine learning systems. The library is 
written in Python 3.8 and is heavily integrated with PyTorch.

For further details about the project, including background information, 
example usage and detailed API documentation, visit 
https://oskarfernlund.github.io/niteshade/.

**Note**: This project is under active development and may be unstable!


Installation
------------

niteshade binaries may be installed from PyPI using ``pip`` 
https://pypi.org/project/niteshade/.

.. code-block:: console

   $ pip install niteshade


Usage
-----

niteshade is a library of functions and classes which allow users to easily 
specify data poisoning attack and defence strategies and simulate their effects 
against online learning using PyTorch models. Detailed information regarding 
the python API and example usage can be found at 
https://oskarfernlund.github.io/niteshade/.


Dependencies
------------

There are two sets of dependencies for this project, both of which can be found 
in the ``env/`` directory in the form of requirements.txt and environment.yml 
files for both ``pip`` and ``Anaconda`` users. The first set are package 
dependencies (pkg_requirements.txt, pkg_environment.yml) which consist 
exclusiveley of the packages required to run the package. These are included 
automatically when niteshade binaries are installed with ``pip``. The second 
set are developer dependencies (dev_requirements.txt, dev_environment.yml), 
which include the package dependencies as well as additional packages required 
for building the documentation, running tests, linting the source code or 
publishing releases.

.. code-block:: console

    $ cd env/

For ``pip`` users:

.. code-block:: console

    # Package dependencies
    $ pip install -r pkg_requirements.txt

    # Developer dependencies
    $ pip install -r dev_requirements.txt

For ``Anaconda`` users:

.. code-block:: console

    # Package dependencies
    $ conda env create -f pkg_environment.yml

    # Developer dependencies
    $ conda env create -f dev_environment.yml


Building the Documentation
--------------------------

To build documentation in various formats, you will need Sphinx and the 
readthedocs theme (included in the developer dependencies). You can build the 
documentation by running ``make <format>`` from the ``docs/`` directory. Run 
``make`` to get a list of all available output formats.

.. code-block:: console

    $ cd docs/
    $ make clean && make <format>


Running Unit and Integration Tests
----------------------------------

This project uses ``pytest`` for unit and integration testing (included in the 
developer dependencies). The tests may be run from the root directory as 
follows:

.. code-block:: console

    $ pytest
    === X passed in x.xx seconds ===


Contributing and Releases
-------------------------

niteshade is an open-source project and contributions are welcome.

Releases are published to PyPI automatically when a tag is pushed to GitHub.

.. code-block:: console

    # Set next version number
    export RELEASE=x.x.x

    # Create tags
    git commit --allow-empty -m "Release $RELEASE"
    git tag -a $RELEASE -m "Version $RELEASE"

    # Push
    git push origin --tags


The Team
--------

niteshade was co-created by Mart Bakler, Oskar Fernlund, Alex Ntemourtsidou, 
Jaime Sabal and Mustafa Saleem in 2022 at Imperial College London. The authors 
may be contacted at the following email addresses:

- Mart Bakler: email 1
- Oskar Fernlund: email 2
- Alex Ntemourtsidou: email 3
- Jaime Sabal: email 4
- Mustafa Saleem: email 5

Big thanks to Dr. Emil C. Lupu for all his feedback and support.

Who will maintain the project after we graduate? Emil?


License
-------

niteshade is covered under the MIT license, as found in the LICENSE file.
