# PFLOTRAN topology optimisation

## Getting started


## Description of the classes

Between each class, user must pass numpy array and assign in place value

### Materials

Methods `mat_properties` and `d_math_properties` to compute the material properties and its derivative from the input density parameter.

* Permeability


### Objectives

Each objective had their own methods to personalize it, but all had a `evaluate` method that return the value of the objective function given the current PFLOTRAN output, and a `d_objective_dX` and `d_objective_dk` for the derivative according to the liquid pressure (`LIQUID_PRESSURE`) and to the permeability (`PERMEABILITY`) output variable, respectively. These methods do not take arguments. If they are needed, they must be set up using `set` method during the initialization.

Objectives are only implemented when design variable is the permeability field. To consider other design varaible, you just have to derive the objective function according to the desired variable.

* Head sum on a domain

Compute the sum of the relative piezometric head in the given region (defined by cell ids). High head could be penalized using a penalizing power (integer).

* Flow through a surface



### Constrains

Each constrains had a method `evaluate` to compute the actual value of the constrains and a method `d_constrain_dp` that return the derivative of the constrain according to the density parameter `p`. Again, these methods do not take arguments. If they are needed, they must be set up using `set` method during the initialization.

* Maximum volume fraction


### PFLOTRAN


### Compute total derivative


### Optimize

