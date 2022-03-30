.. HydrOpTop documentation master file, created by
   sphinx-quickstart on Thu Dec 10 20:03:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HydrOpTop's documentation!
-------------------------------------

Overview
========

HydrOpTop adresses the general problem of finding the optimal design of a system to maximize its performances and submitted to some constrains by the means of topology optimization techniques with the density-based approach in various engineering disciplines.
One may think of what would be the shape of a mechanical part such as its mechanical compliance is being minimized while a volume constraint is applied. 
Or, another wants to maximize the draining capacity of a soil by smarter drain placement.
Another again would like to dissipate as efficiently as possible heat by an optimized heat exchanger.
HydrOpTop permits to solve all of these problems and in these various disciplines by using a unified software. 
The idea behind this library was that topology optimization software generally rely on on-purpose developped solver and on specific syntax, on which user may be reluctant to switch or to learn. 
Using HydrOpTop and with few modifications of their solver codes (if not already carried by a third-person), user could thus used their favorite software to solve topology optimization problems.

HydrOpTop is a Python library which aims to provide a modular, flexible and solver-independant approach for topology optimization (TO) using the density-based approach.
Solver are interfaced through a I/O shield, which allow to define cost function, constraints and filters in a reusable manner for different softwares and codes.
Also, objective functions and constraints are implemented under the same class and without distinction, which means they can be interchanged effortlessly.
Base classes for cost functions/constraints, material parametrizations and filters are also provided so user may define they own TO features with a minimal amount of code.
Therefore, HydrOpTop aims to be the Swiss army knife and a standard exchange place for state-of-the-art tools in TO.

Note, this library is in its first versions, so syntax may change fast and without notice. Once a certain stability will be reach, a first stable version will be released.


Main features
=============

* Distribute two materials (one could be void) in a domain such as an objective function is minimize/maximize.

* Handle any solvers through dedicaced input/output shield.

* Couple any solvers with any objective functions, filters or constraints once written for one and for all.

* Allow users to define well written topology optimization problem in few lines.

* Write results in common and open format to create great graphics and visualization.

* Solve topology optimization problem constrained by one PDE (for instance) as well as various coupled PDE and time dependent problems (in the futur). By extension, any large scale inverse problem can be solved (geophysics, calibration, ...).


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

This page is a first overview on what HydrOpTop and on how to install it.

Installation is first described under the section :doc:`getting_started/installation`.

Then, a small, easy and detailled example is provided in the section :doc:`getting_started/your_first_optimization` to introduce users to the HydrOpTop rudimentary.
Other examples, less detailled however, are also available in :doc:`examples/examples`.

From this, every new HydrOpTop problem can be build using the same structure and by modifying few commands.
Detail of all the available solvers (including their installation), objective functions, filters, materials parametrization and adjoints solving method are described in the :doc:`user_guide/index_user`.

Finally, for those interesting in the HydrOpTop machinery including the implementation of the different adjoints equations and the development of new functions, solvers shields, filters, and so on, the section :doc:`getting_started/how_it_works` is for them.


Index
=====

.. toctree::
  :maxdepth: 1

  getting_started/installation.rst
  getting_started/your_first_optimization.rst
  user_guide/index_user.rst
  examples/examples.rst
  personalization/index_personalization.rst
  getting_started/how_it_works.rst
  topology_optimization_tips.rst
   

Citing
======

Please acknownlegde this github page if you use this software. For instance, it is
a personnaly supported project. 
If it has success, maybe a publication will be written.


Troubleshooting
===============

Please contact
