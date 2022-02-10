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

.. autoclass:: HydrOpTop.IO.IO
  :members:

Input
-----

Simulation results can be 

