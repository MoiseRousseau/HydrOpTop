.. _functions:

Function classes
================



p_Gradient
----------

`p_Gradient` function return a index characterizing whether the material 
`p=1` is placed above material `p=0`:

.. math::
   :label: p_gradient
   
   f = \frac{1}{N} \frac{\sum_{i}^N \max\left(0,(\nabla p_i)_z\right)}
     { \left[ \sum_{i}^N \max\left(0,(\nabla p_i)_z\right)^n \right]^{1/n}}
      - \epsilon

Designed to be used as a constructibility constrain if material 1 could not
be build above material 0 for example. Other gradient direction can also be chosen
The gradient :math:`\nabla p` is  evaluated using the Gauss gradient scheme:

.. math::
   :label: p_gradient_scheme
   
   \nabla p_i = \frac{1}{V_i} 
        \sum_{j \in \partial i} A_{ij} \boldsymbol{n_{ij}} (d_i p_i + (1-d_i)p_j)

Note this method leads rigorously to a second order accurate gradient if and
only if the mesh is non skewed (i.e. the cell center vector intercept the face
exactly at its center), which could not be the case for general unstructured mesh.

Constructor is:

``p_Gradient(direction, tolerance, power)``

where ``direction`` control the ``X``, ``Y`` or ``Z`` direction on which 
calculate the index (default the Z direction), ``tolerance`` the maximum
value of the index (the :math:`\epsilon` value, default is 0.3) and 
``power`` the penalizing power (the `n` value, default is 3).


p_Weighted_Sum_Flux
-------------------

``p_Weighted_Sum_Flux()``


Sum_Flux
--------

Compute the flux through a given surface defined by a list of faces. Faces are
specified by a the two cell ids sharing the face. Fluid is considered incompressible
and with a constant viscosity (i.e. :math:`\rho` and :math:`\mu` are constant). 
Not tested for variably saturated flow.

.. math::
   :label: sum_flux
   
   f = \sum_{(i,j) \in S} \left[A_{ij} \frac{k_{ij}}{\mu} \frac{P_i - P_j + \rho g (z_i - z_j)} {d_{ij}}\right]^n

Constructor is:

``Sum_Flux(connections, option)`` TODO

where ``connections`` is a two dimension array of size (N,2) storing the cell ids 
shared the faces on which to sum the flux. ``option`` argument can take the
following value:

* ``"absolute"``, each face flux are summed in absolute value
* ``"signed"``, each face flux are summed from cell `i` to cell `j`
* ``"signed_reverse"``, each face flux are summed from cell `j` to cell `i`
* ``"squared"``, each face flux are squared (i.e. `n=2`)

Derivative of this function require an adjoint which is set by default, or can
be user supplied using ``func.set_adjoint_problem(adjoint)``.


Sum_Liquid_Piezometric_Head
---------------------------

Constructor is:

``Sum_Liquid_Piezometric_Head()``

Derivative of this function require an adjoint which is set by default, or can
be user supplied using ``func.set_adjoint_problem(adjoint)``

Volume_Percentage
-----------------

The `Volume_Percentage` function compute the ratio of the volume of material
designed by `p=1` on a prescribed domain :math:`D`:

.. math::
   :label: volume_percentage
   
   f = \frac{1}{V_D} \sum_{i \in D} p_i V_i

Constructor is

``Volume_Percentage(cell_ids_to_consider, max_volume)``

where ``cell_ids_to_consider`` is a list of cell ids on which to compute the
volume percentage and ``max_volume`` the maximum volume fraction allowed on the
domain :math:`D` if it is used as a constrain.

