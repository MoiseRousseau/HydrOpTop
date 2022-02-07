.. sensitivity_theory:

Sensitivity calculation theory
==============================



Steady state, one PDE sensitivity
---------------------------------


Transient, one PDE sensitivity
------------------------------



Steady state, two low-coupled PDEs
----------------------------------

Considering a cost function 

.. math::

   f(V(U(X)),U(X),X)

which depends explicitely of the concentration :math:`C(P,X)`
solved by the discretized transport equation 

.. math::

   h(C,P(X),X)
   
which itself depends on the pressure :math:`P` and the material property
:math:`X` solved by the discretized Richard's equation REF, the optimization problem
is given by

.. math::
   
     &\min_X f(C(P,X),X) \\
     &s.t. 
     \begin{array} (
       g(P(X),X) = 0 \\
       h(C(P,X),X) = 0
     \end{array}

The derivative of the cost function relative to the material property 
:math:`X` could be obtained considering the Lagrangian

.. math::
   
   \mathcal{L} = f(C(P,X),P(X),X) + \lambda^T g(P,X) + \mu^T h(C,P(X),X)

Taking the derivative of the lagrangian leads

.. math::
   
   \frac{d \mathcal{L}}{dX} = \frac{df}{dX} = 
   \frac{\partial f}{\partial C} \frac{\partial C}{\partial X} + \frac{\partial f}{\partial X} +
   \lambda^T (\frac{\partial g}{\partial P} \frac{\partial P}{\partial X} + \frac{\partial g}{\partial X}) + 
   \mu^T (\frac{\partial h}{\partial C} \frac{\partial C}{\partial X} + \frac{\partial h}{\partial P}\frac{\partial P}{\partial X} + \frac{\partial h}{\partial X})

Further development leads

.. math::
   
   \frac{df}{dX} = 
   (\frac{\partial f}{\partial C} + \mu^T \frac{\partial h}{\partial C}) \frac{\partial C}{\partial X} +
   (\lambda^T \frac{\partial g}{\partial P} + \mu^T \frac{\partial h}{\partial P}) \frac{\partial P}{\partial X} +
   \lambda^T \frac{\partial g}{\partial X} + \mu^T \frac{\partial h}{\partial X} + 
   \frac{\partial f}{\partial X} 

The unknown terms :math:`\frac{\partial P}{\partial X}` and :math:`\frac{\partial C}{\partial X}` could be withdraw by considering the adjoint equations

.. math::
   
   (\frac{\partial h}{\partial C})^T \mu = - (\frac{\partial f}{\partial C})^T \\
   (\frac{\partial g}{\partial P})^T \lambda = - (\frac{\partial h}{\partial P})^T \mu

And finally, the total derivative of the cost function relative to the material
property :math:`X` is

.. math::
   
   \frac{\partial f}{\partial X} = \lambda^T \frac{\partial g}{\partial X} + 
   \mu^T \frac{\partial h}{\partial X} + \frac{\partial f}{\partial X}
