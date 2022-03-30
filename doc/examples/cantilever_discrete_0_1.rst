.. cantelever_discrete_0_1:

Discrete design with length scale constrain
===========================================

Problem description
-------------------

This example is a variation of the example deals in :doc:`../getting_started/your_first_optimization` with a three fields optimization, i.e. with the raw, the filtered and the projected density parameter fields.
Projected density parameter field is obtained by projecting the filtered field using a smooth Heaviside function.
This strategy could be use to obtain discrete 0-1 design at the end of the optimization by carring a multi-steps optimization process and increasing the steepness of the smooth Heaviside function at each step.


Python code
-----------

The below Python code differ from the original example by the addition of the volume-preserving Heaviside filter, and by the several optimization pass with increasing steepness of the Heaviside function.

.. literalinclude :: ../../examples/Cantilever_discrete/make_cantilever_discrete.py
   :language: python


Results
-------

