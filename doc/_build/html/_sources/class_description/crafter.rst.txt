.. _objectives:

Crafter classes
===============

Crafter
-------

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
 


Available Crafter
-----------------

Then description of the different objectives implemented

Implement yours
---------------

How to 
