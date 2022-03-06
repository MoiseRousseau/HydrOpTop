.. _installation:


Getting started
===============

Installation
------------

HydrOpTop could be installed using Python ``pip`` command:

``pip install HydrOpTop``

Developpement version could be installed clone the HydrOpTop GitHub [repository]()


Solver Installation
-------------------

Shields with the supported solvers are already included, but solvers must be installed or compiled separately.

2D Linear Elasticity Solver
'''''''''''''''''''''''''''

A two-dimensional linear elasticity solver is included with HydrOpTop. 
Source can be find at the following `GitHub repository <https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization>`_.
However, the solver relies on the Eigen library, so it must be installed on your system. 
This can be done with  (On Ubuntu):

.. code-block:: bash
  
  sudo apt install libeigen3 libeigen3-dev

There is no other dependencies.


PFLOTRAN
''''''''

1. Follow instructions `here <https://www.pflotran.org/documentation/user_guide/how_to/installation/linux.html#linux-install>`_ up to the last step (step 5).

2. In the terminal, go to PFLOTRAN source folder:

.. code-block:: bash

  cd pflotran/src/pflotran

3. Checkout a modified version of PFLOTRAN which allow output of sensitivity to solve adjoint (and topology optimization) problems:

.. code-block:: bash

  git checkout moise/make_optimization_v2

4. Build PFLOTRAN with:

.. code-block:: bash

  make pflotran
