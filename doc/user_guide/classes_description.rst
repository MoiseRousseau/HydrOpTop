.. _classes_description:

User guide
==========

Problem setup
-------------

A HydrOpTop optimization problem must contains:
A cost function to minimize
A PFLOTRAN simulation that will return the cost function inputs
A optimization domain parametrization (via the material)
Constrains

Then, all these classes are passed to the problem crafter so that the
internal machinery of HydrOpTop is set.

Classes description
-------------------

.. toctree::
   :maxdepth: 2
   
   functions.rst
   materials.rst
   crafter.rst
   filters.rst
   PFLOTRAN.rst
