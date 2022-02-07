.. _filters:

Filter classes
==============

Smooth or project the density parameter `p` before to convert it to material properties.

Filtering can help the solver to better converge because it can delete 
large material property difference on neighbor cells, and to impose length
constraint on the optimisation for example.


.. autoclass:: HydrOpTop.Filters.Density_Filter

.. figure:: figure/density_filter.png

   Filtering the density parameter according to its nearby value: (a) the 
   density parameter, (b) filtered density parameter using a ball neighbors search, 
   (c) using an ellipsoid neighbors search and (d) increasing the weighting power 
   from 1 to 4.


.. autoclass:: HydrOpTop.Filters.Helmholtz_Density_Filter

.. autoclass:: HydrOpTop.Filters.Heavyside_Filter

.. figure:: figure/heavyside_filter.png

   The three field filtering process: (a) the density parameter (1st field),
   (b) the filtered density parameter with a anisotropic density filter (2nd field), 
   and (c) and (d) the projected density parameter using two heavyside density filter
   with different stepness parameter (3rd field).

.. autoclass:: HydrOpTop.Filters.Volume_Preserving_Heavyside_Filter
