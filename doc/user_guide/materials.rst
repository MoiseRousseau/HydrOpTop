.. _materials:

Parametrization class
=====================

Parametrization class link the (possibly filtered) density parameter to the
simulation material propertie. 
Each parametrization requires the parametrized material property name such as 
defined in the solver IO shield.
A helper plot function also allow visualizing the transformation of the density
parameter to material properties. See the end of the page.

|

.. autoclass:: HydrOpTop.Materials.Identity

|

.. autoclass:: HydrOpTop.Materials.SIMP

|

Similar to the previous parametrization but in logarithm scale

.. autoclass:: HydrOpTop.Materials.Log_SIMP

|

.. autoclass:: HydrOpTop.Materials.RAMP

|

.. automodule:: HydrOpTop.Materials.plot_function
  :members:


