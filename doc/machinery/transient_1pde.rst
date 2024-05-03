.. transient_1pde:

Transient problem, one PDE
==========================

HydrOpTop considers a discrete adjoint.

Considering groundwater flow with a time-dependent pressure field :math:`P` (i.e. head), a design parameter :math:`X` and a general transient cost function

.. math::

    F(P(X),X) = \sum_i^n f(P_i(X),X,t_i)

Where :math:`P_i(X)` is the pressure field at discretized time :math:`t_i` solved by the equation

.. math::

    h(P_{i+1}(X), P_i(X), X, \Delta t_i) = 0

:math:`h` represents the time integrator of the pressure.
The initial pressure field could be the solution to another steady-state PDE (like solution of the Darcy equation)

.. math::

    g(P_0(X), X) = 0


the optimization problem is given by

.. math::
   
    &\min_X F(P(X),X) = \sum_i^n f(P_i(X),X,t_i) \\
    &s.t. 
    \begin{array} (
      g(P_0(X), X) = 0  \\
      h(P_{i+1}(X), P_i(X), X, \Delta t_i) = 0
    \end{array}

The derivative of the cost function relative to the material property 
:math:`X` could be obtained considering the Lagrangian

.. math::

    \mathcal{L} = f(P_0(X),X) + \mu^T g(P_0(X), X) + \sum_{i=1}^n \left[ f(P_i(X),X,t_i) + \lambda_{i}^T h(P_i(X), P_{i-1}(X), X, \Delta t_i) \right] 

Taking the derivative of the lagrangian leads, after few ordering

.. math::
   
    \frac{d \mathcal{L}}{dX} = \frac{dF}{dX} =
    \left(\frac{\partial f}{\partial P_0} + \mu^T \frac{\partial g}{\partial P_0} + \lambda_1^T \frac{\partial h}{\partial P_0} \right) \frac{d P_0}{d X} + 
    \mu^T \frac{\partial g}{\partial X} + \\
    \sum_{i=1}^{n-1} \left[ \left(\frac{\partial f}{\partial P_i} + \lambda_i^T \frac{\partial h}{\partial P_i} \lambda_{i+1}^T \frac{\partial h}{\partial P_i} \right) 
    \frac{\partial P_i}{\partial X} + \lambda_i^T \frac{\partial h}{\partial X} \right] + \\
    \left(\frac{\partial f}{\partial P_n} + \lambda_n^T \frac{\partial h}{\partial P_n} \right) \frac{d P_n}{d X} + \lambda_n^T \frac{\partial h}{\partial X} + n \frac{\partial f}{\partial X}

The unknown terms :math:`\frac{\partial P_j}{\partial X}` could be withdraw by considering the adjoint equations

.. math::
    
    (\frac{\partial h}{\partial P_n})^T \lambda_n = - (\frac{\partial f}{\partial P_n})^T \\
    (\frac{\partial h}{\partial P_i})^T (\lambda_i + \lambda_{i+1}) = - (\frac{\partial f}{\partial P_i})^T \\
    (\frac{\partial g}{\partial P_0})^T \mu = - (\frac{\partial f}{\partial P_0})^T - (\frac{\partial h}{\partial P_0})^T \lambda_1

And finally, the total derivative of the cost function relative to the material
property :math:`X` is

.. math::
   
   \frac{dF}{dX} = n \frac{\partial f}{\partial X} + \mu^T \frac{\partial g}{\partial X} +
   \sum_{i=1}^{n} \lambda_i^T \frac{\partial h}{\partial X}

Yet, all of this was not yet tested...   
