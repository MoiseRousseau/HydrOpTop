.. _io:

Input/Output
============

The Input/Output class is responsible to save and interact with optimization 
results.

``define_output_file()``: Define the file where to save the optimization 
results.

``define_output_format()``: Set the output format. Available format:
* XDMF (defaut)
* VTU
* Medit

``define_output_log()``: Set the output file where the history of the cost 
function and constrain are stored.

``no_output_initial()``: Does not save the initial distribution of the density 
parameter.

``output_every_iteration(n)``: Output the density parameter :math:`p` every
:math:`n` iteration in file specified by the `out` argument and in HDF5 format.
Description of the HDF5 format TODO

``output_gradient()``: Output the gradient of the objective function relative to
the density parameter :math:`p`. 

``output_gradient_constrain()``: Output the gradient of the constrains relative to
the density parameter :math:`p`.

