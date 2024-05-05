.. HydrOpTop documentation master file, created by
   sphinx-quickstart on Thu Dec 10 20:03:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HydrOpTop's documentation!
-------------------------------------

Overview
========

HydrOpTop adresses the general problem of finding the optimal design of a hydrogeological system to maximize its performances and submitted to some constrains by the means of topology optimization technics with the density-based approach as used in various engineering disciplines.
One may think of maximizing the draining capacity of a soil by smarter drain placement.
Another would like to calibrate permeability of a soil to some data.


Main features
=============

* Large-scale ydrogeological inverse problems
* Steady-state adjoint solver (time-dependent in the future)
* Permeability calibration
* Man-made structure design optimisation
* Fault placements (in the future)


Solvers interfaced
==================

List of solvers interfaced with HydrOpTop:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Solver name
     - Description
     - Comments
   * - `PFLOTRAN <https://www.pflotran.org>`_
     - Subsurface reactive transport code
     - Only Richards mode supported
   * - `MinimalFEM <https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization>`_
     - 2D mechanical linear elasticity
     - N/A


How to read this documentation ?
================================

Installation is first described under the section :doc:`installation`.

Examples are available in :doc:`gallery_examples/index`.
They can be used as template for your optimisation problem or for helping you getting started


Detail of all HydrOpTop object, such as objective functions, filters, materials parametrization and adjoints solving are described in the :doc:`user_guide/index_user`.

For those interesting in the HydrOpTop machinery including the implementation of the different adjoints equations, see :doc:`machinery/how_it_works`.

Finally, for the development of new functions, solvers shields, filters, and so on, the section :doc:`personalization` is for them.


Index
=====

.. toctree::
  :maxdepth: 1

  installation.rst
  gallery_examples/index
  user_guide/index_user.rst
  machinery/how_it_works.rst
  personalization.rst
   

Citing
======

* Rousseau, M., Pabst, T. Topology optimization of in-pit codisposal of waste rocks and tailings to reduce advective contaminant transport to the environment. Struct Multidisc Optim 65, 168 (2022). https://doi.org/10.1007/s00158-022-03266-1

Please also star the GitHub `repository <https://github.com/MoiseRousseau/HydrOpTop>`_ if you use this software. 


Troubleshooting
===============

Please open an issue on the GitHub repository.
