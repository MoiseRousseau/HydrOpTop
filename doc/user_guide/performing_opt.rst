.. _performing_opt:

HydrOpTop basics
=================

NOT UP TO DATE, can be unreliable.

Problem Setup
-------------

HydrOpTop solves topology optimization problem from a input ASCII (text) file containing Python command.
This file contains all the features of your optimization problem (function to optimize, constraints, optimizer and so on).
In detail, this file must contains several part each specifying a part of your problem:

* A cost function to minimize/maximize
* A solver whose HydrOpTop will interact with
* The optimization problems constraints 
* Material parametrization that relate the topology optimization density parameter to the material parameters

Then, all these different part are passed to the problem crafter (see below) so that a HydrOpTop optimization problem is set and returned to the user.

Using the returned object, user can now specify the desired output behavior, file and format, and finally using the ``solve`` method with appropriate argument to solve the topology optimization problem (see below).


The Crafter Class
-----------------

Describe crafter




Input and Output
----------------

Describe IO


Performing Optimization
-----------------------

Optimizer, parameter

Saving file, and run Python


Results
-------

What does HydrOpTop return ?



