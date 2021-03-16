.. _objectives:

Material classes
=================

Permeability
------------

`perm = Permeability([min_k, max_k], cell_ids_to_parametrize, power)`: Create a
SIMP parametrization of the permeability in the range given by `[min_k, max_k]`
on the cell ids `cell_ids_to_parametrize` and with the given penalization power
`power`. By default, parametrization is carried on all the cell in PFLOTRAN 
mesh and with `power=3`.

`perm.plot_K_vs_p()`: Plot the permeability and its derivative as a function of
the density parameter :math:`p`. Require the `matplotlib` Python library to 
be installed.
