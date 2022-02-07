.. _functions:

Objective functions and constraints
===================================

Note in HydrOpTop, functions and constraints inherit from the same class.
By doing so, they can easily be exchanged.

Classed by alphabetical order.


Generic objective functions
'''''''''''''''''''''''''''

.. autoclass:: HydrOpTop.Functions.p_Gradient

.. autoclass:: HydrOpTop.Functions.Sum_Variable

.. autoclass:: HydrOpTop.Functions.Volume_Percentage


Linear elasticity functions
'''''''''''''''''''''''''''

.. autoclass:: HydrOpTop.Functions.Mechanical_Compliance


PFLOTRAN specific functions
'''''''''''''''''''''''''''

.. autoclass:: HydrOpTop.Functions.Head_Gradient

.. autoclass:: HydrOpTop.Functions.Mean_Liquid_Piezometric_Head

.. autoclass:: HydrOpTop.Functions.p_Weighted_Head_Gradient

.. autoclass:: HydrOpTop.Functions.Sum_Flux

.. autoclass:: HydrOpTop.Functions.p_Weighted_Sum_Flux

