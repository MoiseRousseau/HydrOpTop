.. _filters:

Filter classes
==============

A very comprehensive review of filters for density-based topoly optimization
in provided in the paper `Volume preserving nonlinear density filter based 
on heaviside functions` by Xu, Cai and Cheng (2009).


Density filter
--------------

Smooth the density parameter `p` according to its nearby value:

.. math::
  :label: density_filter
  
  \bar{p_i} = \frac{R V_i p_i + \sum_{j \in \partial i} (R-d_{ij}) V_j p_j}
                    {R V_i + \sum_{j \in \partial i} (R-d_{ij}) V_j}
  
This filter was proposed by Bruns and Tortorelli (2001) and Bourdin (2001). 

