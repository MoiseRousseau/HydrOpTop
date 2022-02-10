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

0. Install ``HydrOpTop`` if not already done (see above).

1. Install the required Eigen library header (On Ubuntu):

.. code-block:: bash
  
  sudo apt install libeigen3-dev

2. Open a Python interpretor and run:

.. code-block:: python
  
  import HydrOpTop.Solvers.Linear_Elasticity_2D as solver
  solver.__get_and_compile_solver__()

This will automatically reach the `GitHub repository <https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization>`_ where solver is located, and launch the compilation.
Solver executable will be located in HydrOpTop installation directory (Python cache).


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
