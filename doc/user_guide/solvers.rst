.. _solvers:

Solvers
=======

This section provide information on own to install solver interfaced with HydrOpTop and how to use the specific input/output shield.
These shield allow HydrOpTop to communicate parametrized cell properties to the solvers, and get back the solved variables to compute the cost function and constraints, the sensitivities to solve the adjoint problem and get the objective function derivative, and sometimes some quantity related to the mesh.


PFLOTRAN
--------

PFLOTRAN is a XXX

Solver installation and preparation for HydrOpTop
'''''''''''''''''''''''''''''''''''''''''''''''''


Interact with the solver
''''''''''''''''''''''''

Constructor is:

``sim = PFLOTRAN(pflotranin, mesh_info=None)``

with ``pflotranin`` the name of the PFLOTRAN input file (mandatory), and
``mesh_info`` the path to a PFLOTRAN output file containing simulation
independant PFLOTRAN output variable such as the mesh informations (face area or
cell volume for example).
Providing mesh information can help reduce the size of PFLOTRAN output file
at every iteration, therefore saving time and increase the life of your SSD!

The below PFLOTRAN specific methods are available:

``sim.set_parallel_calling_command(n,command)``: Set the number of process 
:math:`n` to run and the command to call (default is `mpiexec.mpich`)

``sim.get_region_ids(name)``: Return a numpy 1D array containing the ID of the cell
belonging to the region `name` as in the PFLOTRAN input file.




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


