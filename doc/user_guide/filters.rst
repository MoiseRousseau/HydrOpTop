.. _filters:

Filter classes
==============

Smooth the density parameter `p` before to convert it to material properties.

Filtering can help the solver to better converge because it can delete 
large material property difference on neighbor cells, and to impose length
constrain on the optimisation.


Density filter
--------------

Smooth the density parameter `p` at a given cell `i` according to its value
at the neighboring cells `j` weighted by the distance of their respective 
centers:

.. math::
  :label: density_filter
  
  \bar{p}_i = \frac{R V_i p_i + \sum_{j \in \partial i} (R-d_{ij}) V_j p_j}
                    {R V_i + \sum_{j \in \partial i} (R-d_{ij}) V_j}
  
This filter was proposed by Bruns and Tortorelli (2001) and Bourdin (2001). 
Also see Xu et al. (2009).

Constructor is:

``Density_Filter(filter_radius)``

where the argument ``filter_radius`` is the ball radius on which to search 
for neighboring cell center for averaging the density parameter. When a list
is provided (i.e. ``filter_radius=[dx,dy,dz]``), the cell centers are searched 
into a ellipsoid of half axis dx, dy and dz.


Heavyside density filter
------------------------

Apply the smooth Heavyside function to the density parameter with a given
steepness and cutoff according to (Xu et al. 2009):

.. math::
  :label: heavyside_density_filter
  
  \tilde{p}_i = \left\{ 
            \begin{array}{ll}
       \eta \left[ e^{-\beta(1-\bar{p}_i/\eta)} - 
         (1-\frac{\bar{p}_i}{\eta}) e^{-\beta}\right] \quad \mbox{if} \quad \bar{p}_i<\eta \\
      (1-\eta) \left[ 1-e^{-\beta(\bar{p}_i-\eta)/(1-\eta)} + 
         \frac{\bar{p}_i-\eta}{1-\eta}e^{-\beta} \right] + \eta \quad \mbox{else}
            \end{array} \\
                \right.


Constructor is:

``Heavyside_Density_Filter(base_density_filter, cutoff, steepness)``

``cutoff`` is the cutoff parameter :math:`\eta` (i.e. the value of 
:math:`p_i` where the step is located) and ``steepness`` the steepness
of the smooth Heavyside function :math:`\beta`. 
``base_density_filter`` is another density filter on whom apply the 
Heavyside filter, such as a basic ``Density_Filter(filter_radius)``.

Be careful with the steepness parameter: a too high value (i.e. :math:`\beta>5`) 
may cause the derivative to be close to 0 everywhere away from the step, causing optimization not to converge and stay close to the initial density parameter 
provided.
This filter was proposed by Xu et al. (2009). See the original publication
for more detail.

Add a figure here.


Reference
------------------------

Xu, Cai and Cheng (2009)

