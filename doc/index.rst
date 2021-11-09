.. HydrOpTop documentation master file, created by
   sphinx-quickstart on Thu Dec 10 20:03:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HydrOpTop's documentation!
=====================================


Overview
========

HydrOpTop aimed at provide tools to perform topology optimization for 
geoenvironmental engineering.  It adresses the general problem of finding the
optimal design of a system to maximize its performances and submitted to some
constrains. One may think of where to place a reactive barrier near a 
contamined site to maximize the recuperation of a contaminant. Or, another
wants to maximize the draining capacity of a soil by smarter drain placement.
HydrOpTop permitted to solve both of these problems using topology 
optimization. 

HydrOpTop consists of several Python classes, ready-made functions to optimize
or to use as constrains, and preprogrammed material parametrizations that 
allow the user to perform topology optimization effortlessly. 
It can run optimization problems consisting of saturated as well as 
unsaturated flow, with solute transport and both in steady-state or transient
conditions.
It is build upon popular Python packages such as numpy and scipy to run topology optimization, while the hydrogeological solver uses the finite volume flow 
and transport code PFLOTRAN (see X). 

Installation
============

HydrOpTop could be installed using Python3 ``pip3`` command:

``pip3 install hydroptop``

Note to be able to run optimization problems, PFLOTRAN should be installed
on your system which should be called by a terminal command named ``pflotran``.


Citing
======
If you are using HydrOpTop in a scientic publication, please consider citing
the following reference:

Rousseau Moise (2021), HydrOpTop: a Python package to solve topology
optimization problems in geoenvironmental engineering.


Index
=====

.. toctree::
   :maxdepth: 1
   
   getting_started/index_started.rst
   user_guide/index_user.rst
   machinery/index_machinery.rst
   examples/examples.rst
   personalization/index_personalization.rst
   troubleshoot/troubleshoot.rst
