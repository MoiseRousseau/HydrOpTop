.. HydrOpTop documentation master file, created by
   sphinx-quickstart on Thu Dec 10 20:03:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HydrOpTop's documentation!
-------------------------------------

Overview
========

HydrOpTop is a Python library which aims to provide a modular, flexible and solver-independant approach for topology optimization (TO) using the densoty-based approach.
Solver are interfaced using a I/O shield, which thus allow to define cost function, constraints and filters in a reusable manner for different softwares and codes.
Also, objective functions and constraints are implemented under the same class and without distinction, which means they can be interchanged effortlessly.
Base classes for cost functions/constraints, material parametrizations and filters are also provided so user may define they own TO features with a minimal amount of code.
Therefore, HydrOpTop aims to be the Swiss army knife and a standard exchange place for state-of-the-art tools in TO.


Purpose
=======

..
  One may think of what would be the shape of a mechanical part such as its mechanical compliance is being minimized while a volume constraint is applied. 
  Or, another wants to maximize the draining capacity of a soil by smarter drain placement.
  Another again would like to dissipate as efficiently as possible heat by an optimized heat exchanger.
  HydrOpTop permits to solve all of these problems and in these various disciplines by using a unified software.

HydrOpTop adresses the general problem of finding the optimal design of a system to maximize its performances and submitted to some constrains by the means of topology optimization techniques with the density-based approach in various engineering disciplines using a .
It aimed to create a unified software to perform topology optimization approach regarless of the phenomenon studied, the software used and the coding abilities of users, and get them familiar to state of art approach. 
The idea behind this library was that topology optimization software generally rely on on-purpose developped solver and on specific syntax, on which user may be reluctant to switch or to learn. 
Using HydrOpTop and with few modification of their solver codes (if not already carried by a third-person), user could thus used their favorite software to perform topology optimization problems.
HydrOpTop name comes from the Hydra in the greek mythology where each head can correspond to one simulation code.

HydrOpTop consists of several Python classes, ready-made functions to optimize
or to use as constrains, preprogrammed material parametrizations and optimization algortihms that 
allow the user to perform topology optimization effortlessly. 
It can also operate with many different solvers through solver-specific shields.
Example integrations of new solver in the HydrOpTop framework are provided in the examples section.
For instance, a finite element 2-dimensional linear elasticity solver is integrated, along with the subsurface reactive transport finite volume code PFLOTRAN (see :doc:`user_guide/solvers` section in user guide).

..
  Do I need to know python ?


What HydrOpTop can do ?
=======================

Distribute one or several material in a domain such as an objective is reached.

Handle any solver through dedicaced input/output shell solver specific (but which need to be written)

Handle any objective function, filters or constraints predefined or written by the user.

Allow user to define well written tpology optimization problem in few lines

Write results in open format which allow creating great graphics

Solve topology optimization problem constrained by one PDE as well as various coupled PDE, time dependent problem

By extension, any large scale inverse problem can be solved (geophysics, calibration, ...)


What HydrOpTop cannot do ?
==========================

It is not a miraculous software which you design your solution for you.

..
  How to read this documentation ?
  ================================

  This page is a first overview on what HydrOpTop and on how to install it.

  Installation is first described under the section :doc:`getting_started/installation`.

  <!--For those which even doesn't know what is design or topology optimization, or which want a quick introduction or a reminder, the page :doc:`getting_started/topology_optimization` is for you.-->

  Then, a small, easy and detailled example is provided in the section :doc:`getting_started/your_first_optimization` to introduce users to the HydrOpTop rudimentary.
  Other examples, less detailled however, are also available in :doc:`examples/examples`.

  From this, every new HydrOpTop problem can be build using the same structure and by modifying few commands.
  Detail of all the available solvers (including their installation), objective functions, filters, materials parametrization and adjoints solving method are described in the :doc:`user_guide/index_user`.

  Finally, for those interesting in the HydrOpTop machinery including the implementation of the different adjoints equations and the development of new functions, solvers shields, filters, and so on, the section :doc:`machinery/index_machinery` is for them.


Index
=====

.. toctree::
  :maxdepth: 1

  getting_started/installation.rst
  getting_started/how_it_works.rst
  getting_started/your_first_optimization.rst
  user_guide/index_user.rst
  examples/examples.rst
  personalization/index_personalization.rst
  topology_optimization_tips.rst
   

Citing
======

Please acknownlegde this github page if you use this software. For instance, it is
a personnaly supported project. 
If it has success, maybe a publication will be written.


Troubleshooting
===============

Please contact
