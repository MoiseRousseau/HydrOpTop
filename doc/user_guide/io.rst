.. _io:

Input and Output
================

The Input/Output (IO) class is responsible to save and interact with optimization 
results.
Resuls saved include the history of the cost function and the constraints values 
the simulation mesh, the raw density parameter :math:`p`, the filtered density 
parameter :math:`\bar{p}` or solver input or output variables.

IO class is a properties of the crafter and is assessed through ``crafter.IO``.


Output
------

The following methods can be used to specify the optimization logs and results.

``define_output_file(filename)``: Define the file name (without its extension) where to save the optimization 
results (default: ``filename="HydrOpTop"``)

``define_output_format(format)``: Set the output format. Available format:

* XDMF (``xdmf``, not functional)
* VTU (``vtu``, defaut)
* Medit (``mesh``, not functional)


``define_output_log(filename)``: Set the output file where the history of the cost 
function value and constraints are stored. By default, the same filename as defined with the command ``define_output_file()``.

``no_output_initial()``: Does not save the initial distribution of the density 
parameter (default is enabled).

``output_every_iteration(n)``: Specify to output the density parameter :math:`p` every 
:math:`n` iteration in file and format specified in ``define_output_file()`` and ``define_output_format()`` methods.

``output_gradient()``: Enable writing the gradient of the objective function relative to
the density parameter :math:`p` in the output file.

``output_gradient_constraints()``: Enable writing the gradient of the constraints relative to
the density parameter :math:`p`.

``write_field_to_file(X, filename, format_, at_ids, Xname)``: Write the simulation
related fields ``X`` (numpy array or list of numpy array) in file ``filename``
in format ``format_`` (optional) under the field ``Xname`` (optional). ``X``
can be defined on a subset of the simulation mesh at cell ids defined
by ``at_ids`` variable.


Input
-----

Simulation results can be 

