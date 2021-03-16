.. _crafters:

Crafter classes
===============

The Crafter class is responsible to build the topology optimization problem,
to make the link between the inputs and the outputs of the material property
parametrization, PFLOTRAN, the objective function, the constrains and the
different filters, as well as providing already-made function to be called by
popular optimizers (such as `nlopt` for example).

It first take the cell ids to parametrize specified in material property and
create one independant variable for each cell, stored in a vector writed `p`,
which represent the SIMP density parameter. Correspondance between `p` and the
cell ids in PFLOTRAN simulation is also stored (i.e. `p[0]` linked to cell id
number X, `p[1]` linked to Y, ...)

Steady State Crafter
--------------------

``Steady_State_Crafter(objective, solver, mat_props, constrains, filter)``

The following method is associated with the Steady State Crafter

``get_problem_size()``: Return the optimization problem size (i.e. the degre of
freedom of the problem).

``do_not_run_simulation()``: Optimization will be run normally but no call to
PFLOTRAN will be done. Usefull to debug input script or functions if 
PFLOTRAN output files are already in the folder for example.

``output_every_iteration(n, out)``: Output the density parameter :math:`p` every
:math:`n` iteration in file specified by the `out` argument and in HDF5 format.
Description of the HDF5 format TODO

``output_gradient()``: Output the gradient of the objective function relative to
the density parameter :math:`p`. Output synchronized with `output_every_iteration()`
and writed in the same output file.

``nlopt_function_to_optimize(p, grad)``: Wrapper function to pass to ``nlopt``
optimizer.

``nlopt_constrain(i)``: Return a wrapper function to pass to ``nlopt``
optimizer corresponding to the ith constrains.
