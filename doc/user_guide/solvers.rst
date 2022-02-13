.. _solvers:

Solvers
=======

This section provide information on own to install solver interfaced with HydrOpTop and how to use the specific input/output shield.
These shield allow HydrOpTop to communicate parametrized cell properties to the solvers, and get back the solved variables to compute the cost function and constraints, the sensitivities to solve the adjoint problem and get the objective function derivative, and sometimes some quantity related to the mesh.


PFLOTRAN
--------

PFLOTRAN is a subsurface reactive transport code.

For instance only steady state processes in Richards mode are supported.


.. autoclass:: HydrOpTop.Solvers.PFLOTRAN
  :members:


Two-Dimensional Linear Elasticity
---------------------------------

The Two-Dimensional Linear Elasticity is a linear mechanical solver which solve the displacement and the Von-Misses stress giving a 2D triangular unstructured grid on which each triangle can have a different Young modulus but a constant poisson ratio over the grid.
The solver is available at X, which is a modified version of those from X to enable writing sensitivities and handle heterogeneous Young modulus.

Installation
''''''''''''

Located in source. Must be compiled before use.
Call by the command ``MinimalFEM``.
Input need a 2D mesh (``.mesh`` file), material property file (``.matprops``) and a boundary condition (``.bcs``).


Interact with the solver
''''''''''''''''''''''''




Command common to all solvers
-----------------------------

TODO


