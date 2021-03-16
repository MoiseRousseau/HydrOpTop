.. _theory:

Theory guide
============

SIMP material parametrization
-----------------------------


Steady state flow adjoint sensitivity
-------------------------------------



Steady state transport adjoint sensitivity
------------------------------------------

Considering a cost function 

.. math::
   :label: cost_function_steady_transport
   
   f(C(P,X),X)

which depends explicitely of the concentration :math:`C(P,X)`
solved by the discretized transport equation 

.. math::
   :label: transport_equation
   
   h(C,P(X),X)
   
which itself depends on the pressure :math:`P` and the material property
:math:`X` solved by the discretized Richard's equation REF, the optimization problem
is given by

.. math::
   :label: opt_problem_steady_transport
   
     &\min_X f(C(P,X),X) \\
     &s.t. 
     \begin{array} (
       g(P(X),X) = 0 \\
       h(C(P,X),X) = 0
     \end{array}

The derivative of the cost function relative to the material property 
:math:`X` could be obtained considering the Lagrangian

.. math::
   :label: lagrangian_steady_transport
   
   \mathcal{L} = f(C(P,X),P(X),X) + \lambda^T g(P,X) + \mu^T h(C,P(X),X)

Taking the derivative of the lagrangian leads

.. math::
   :label: d_lagrangian_steady_transport
   
   \frac{d \mathcal{L}}{dX} = \frac{df}{dX} = 
   \frac{\partial f}{\partial C} \frac{\partial C}{\partial X} + \frac{\partial f}{\partial X} +
   \lambda^T (\frac{\partial g}{\partial P} \frac{\partial P}{\partial X} + \frac{\partial g}{\partial X}) + 
   \mu^T (\frac{\partial h}{\partial C} \frac{\partial C}{\partial X} + \frac{\partial h}{\partial P}\frac{\partial P}{\partial X} + \frac{\partial h}{\partial X})

Further development leads

.. math::
   :label: d_lagrangian_steady_transport_2
   
   \frac{df}{dX} = 
   (\frac{\partial f}{\partial C} + \mu^T \frac{\partial h}{\partial C}) \frac{\partial C}{\partial X} +
   (\lambda^T \frac{\partial g}{\partial P} + \mu^T \frac{\partial h}{\partial P}) \frac{\partial P}{\partial X} +
   \lambda^T \frac{\partial g}{\partial X} + \mu^T \frac{\partial h}{\partial X} + 
   \frac{\partial f}{\partial X} 

The unknown terms :math:`\frac{\partial P}{\partial X}` and :math:`\frac{\partial C}{\partial X}` could be withdraw by considering the adjoint equations

.. math::
   :label: adjoint_equations_steady_transport
   
   (\frac{\partial h}{\partial C})^T \mu = - (\frac{\partial f}{\partial C})^T \\
   (\frac{\partial g}{\partial P})^T \lambda = - (\frac{\partial h}{\partial P})^T \mu

And finally, the total derivative of the cost function relative to the material
property :math:`X` is

.. math::
   :label: total_derivative_steady_transport
   
   \frac{\partial f}{\partial X} = \lambda^T \frac{\partial g}{\partial X} + 
   \mu^T \frac{\partial h}{\partial X} + \frac{\partial f}{\partial X}
