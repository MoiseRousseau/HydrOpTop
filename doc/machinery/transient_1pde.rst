.. transient_1pde:

Transient problem, one PDE
=====================

Considering groundwater flow with a  pressure field :math:`P` (i.e. head), a design parameter :math:`X` and a general cost function

.. math::

   f(P(X),X)

:math:`P(X)` solved by the Darcy / Richards equation

.. math::

   g(P(X),X) = 0

the optimization problem is given by

.. math::
   
     &\min_X f(C(P,X),X) \\
     &s.t. 
     h(C(P,X),X) = 0

todo
