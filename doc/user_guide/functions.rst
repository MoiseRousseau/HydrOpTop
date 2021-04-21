.. _functions:

Function classes
================

Classed by alphabetical order.

p_Gradient
----------

`p_Gradient` function return a index characterizing whether the material 
`p=1` is placed above material `p=0`:

.. math::
   :label: p_gradient
   
   f = \frac{1}{V_D} \sum_{i \in D} V_i \max\left(0,(\nabla p_i)_c\right)^n
          - \epsilon

Designed to be used as a constructibility constrain if material 1 could not
be build above material 0 for example.
The :math:`\max()` function is represented through a smooth Heavyside function.
The gradient :math:`\nabla p` is  evaluated using the Gauss gradient scheme:

.. math::
   :label: p_gradient_scheme
   
   \nabla p_i = \frac{1}{V_i} 
        \sum_{j \in \partial i} A_{ij} \boldsymbol{n_{ij}} 
          \left\{ 
            \begin{array}{ll}
              p_j \mbox{ if } z_i > z_j \\
              p_i \mbox{ else}
            \end{array} \\
          \right.

Note this method leads rigorously to a second order accurate gradient if and
only if the mesh is non skewed (i.e. the cell center vector intercept the face
exactly at its center), which could not be the case for general unstructured mesh.
For such a case, gradient can be corrected by substracting the gradient considering
:math:`p=1` on all the domain and weighted by `p`:

.. math::
   :label: p_gradient_correction
   
   (\nabla p_i)_c = \nabla p_i - p_i (\nabla p_i)_{p=1} 

The corrected gradient does not show skewness error when :math:`p` is constant
for all the cell neighbors.

Constructor is:

``p_Gradient(direction, tolerance, power, correction)``

where ``direction`` control the ``X``, ``Y`` or ``Z`` direction on which 
calculate the index (default the Z direction), ``tolerance`` the maximum
value of the index (the :math:`\epsilon` value, default is 0.3), 
``power`` the penalizing power (the `n` value, default is 3) and
``correction`` a boolean to enable the correction as decribed above.

Require the PFLOTRAN outputs ``FACE_AREA``, ``VOLUME``, 
``FACE_CELL_CENTER_VECTOR_{direction}`` and ``PRINT_CONNECTION_IDS``.


p_Weighted_Sum_Flux
-------------------

`p_Weighted_Sum_Flux` return a number characterizing the total flowrate in
material designed by `p=1` in the considered cell.
In practice, it could be used to minimize the mean flux in material designed 
by `p=1`.

`p_Weighted_Sum_Flux` returned value is defined as the sum of the squared flux
through each connection of each considered cell and weighted by the cell 
material parameter:

.. math::
   :label: p_weighted_sum_flux
   
   f = \sum_{i \in D} p_i \sum_{j \in \partial i} A_{ij} K_{ij} (P_j - P_i - \rho g {z_j-z_i})

Constructor is:

``p_Weighted_Sum_Flux(cell_ids_to_consider=None, invert_weighting=False)``

where ``cell_ids_to_consider`` is a list of the cell to sum the 
flowrate on and ``invert_weighting`` a boolean to invert the weighting and 
rather consider the flux in the material given by `p=0` (i.e. 
:math:`p'=1-p`).

Require the PFLOTRAN outputs ``LIQUID_PRESSURE``, ``FACE_AREA``, 
``PERMEABILITY``, ``FACE_UPWIND_FRACTION``, ``FACE_DISTANCE_BETWEEN_CENTER``, 
``Z_COORDINATE`` and ``CONNECTION_IDS``.


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

Require the PFLOTRAN outputs ``LIQUID_PRESSURE``, ``FACE_AREA``, 
``PERMEABILITY``, ``FACE_UPWIND_FRACTION``, ``FACE_DISTANCE_BETWEEN_CENTER``, 
``Z_COORDINATE`` and ``CONNECTION_IDS``.


Mean_Liquid_Piezometric_Head
----------------------------

The `Mean_Liquid_Piezometric_Head` function compute the mean of the piezometric
head in the given cell ids:

.. math::
   :label: mean_liquid_pz_head
   
   f = \frac{1}{V_D} \sum_{i \in D} V_i (\frac{P-P_{ref}}{\rho g} + z_i)
   
Constructor is:

``Mean_Liquid_Piezometric_Head()``

Derivative of this function require an adjoint which is set by default, or can
be user supplied using ``func.set_adjoint_problem(adjoint)``.

Require the PFLOTRAN output variable ``LIQUID_PRESSURE``.


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

Require the PFLOTRAN output variable ``VOLUME``.
