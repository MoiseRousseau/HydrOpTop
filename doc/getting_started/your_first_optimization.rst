.. _your_first_optimization:

A first simple optimization using HydrOpTop
===========================================

Problem description
-------------------

Conceptual and numerical models
'''''''''''''''''''''''''''''''

Your first topology optimization problem consists in a classical benchmark test in linear mechanic, that is the cantilever. 
For this example, the 2D linear elasticity solver is used. 
The design domain is a rectangle of 5 units high and 10 units wide discretized with triangles, where the cantilever is fixed at the left boundary and the load (unitary) applied at right bottom corner.

A numerical model of the above is built using a finite element solver and a triangular mesh.
The finite element solver chosen is those already compiled and shipped with HydrOpTop libraries (available alone `here <https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization>`_.
DESCRIBE THE INPUT FILE.
The triangular mesh was generated using `Salome CAD <www.salome-platform.org>`_ software.


Topology Optimization Problem
'''''''''''''''''''''''''''''

The topology optimization is carried out by considering a parametrization of the Young modulus in each mesh triangle using a SIMP parametrization with a penalization power of 3.
A ball density filter with a radius of  is also applied to avoid the formation of checkerboard pattern and to apply a minimal size constraint of 0.3 units.
A volume constraint of 50% is considered.

The Python Code
---------------

Import first the necessary libraries (i.e. ``NumPy`` and ``HydrOpTop`` components):

.. code-block:: python

  import numpy as np
                                    
  from HydrOpTop.Functions import Mechanical_Compliance, Volume_Percentage
  from HydrOpTop.Materials import SIMP
  from HydrOpTop.Filters import Density_Filter
  from HydrOpTop.Crafter import Steady_State_Crafter
  from HydrOpTop.Solvers import Linear_Elasticity_2D

Then, create a solver object with the 2D linear elasticity solver included in HydrOpTop and specify the appropriate simulation control file prefix for the solver (i.e. the ``cantilever.XXX`` file):

.. code-block:: python

  sim = Linear_Elasticity_2D("cantilever")

Parametrize the cell of the whole mesh using a SIMP parametrization of the Young modulus:

.. code-block:: python

  young_modulus = SIMP(cell_ids_to_parametrize="all", 
                       property_name="YOUNG_MODULUS", 
                       bounds=[0, 2000], 
                       power=3)

In the above, the ``cell_ids_to_parametrize`` argument specify which cell ids to parametrize (set to ``"all"`` because optimization domain span all the mesh), while the ``property_name`` argument specify the cell property to parametrize.
Also, the ``bounds`` argument describe the bound of the parametrization, i.e. the Young modulus is 0 when the density parameter ``p=0``, and 2000 MPa if ``p=1``.

Define the cost function (mechanical compliance) on the next line using the dedicaced class:

.. code-block:: python

  cf = Mechanical_Compliance(ids_to_consider="everywhere")
  
``ids_to_consider`` argument tells HydrOpTop on which elements to calculate the compliance.
Since the whole domain is considered here, the argument is set to ``"everywhere"``.

Define the volume percentage constraint with:

.. code-block:: python

  max_vol = Volume_Percentage("parametrized_cell", 0.5)
  
Create the density filter and its projection using:

.. code-block:: python

  dfilter = Density_Filter(0.3)
  hfilter = Volume_Preserving_Heavyside_Filter(0.5, 1, max_vol) #cutoff, stepness

At this stage, all the different components of the optimization problem are now defined.
The optimization problem can be now be crafted and stored into the ``crafted_problem`` variable:

.. code-block:: python

  crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], filters=[dfilter, hfilter])
  crafted_problem.IO.output_every_iteration(2)
  crafted_problem.IO.define_output_format("vtu")

In the above, TODO.
The two last line specify HydrOpTop to output the variable related to the optimization (i.e. the density parameter, raw and projected) every two iterations in ``vtu`` format to be latter visualize in Paraview and create a nice animation.

The last step is to specify the initial density parameter field as a numpy array.
For this example, a homogeneous initial density parameter (``p=0.2``) is chosen:

.. code-block:: python

  p_ini = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2

The topology optimization problem can now be solved! This is done calling the following function:

.. code-block:: python

  out = crafted_problem.optimize(optimizer="nlopt-mma", 
                                 action="minimize", 
                                 max_it=20, 
                                 ftol=0.0001, 
                                 initial_guess=p_ini)

The above specify to use the MMA algorithm from the library ``nlopt`` to minimize the cost function, until a maximum of 20 iterations is reached or when the relative variation of the cost function is below 0.001, and using the homogeneous initial density parameter of 0.2 everywhere.
The result is stored in the ``out`` variable.

.. code-block:: python

  crafted_problem.IO.write_field_to_file(out.p_filtered_opt, "Filtered_density", "./out.mesh")

Over! This code consists in 20 lines of meaningfull Python!

Results
-------

Add picture to make a great introduction
