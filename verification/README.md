# PFLOTRAN topology optimization verification

This folder aimed to verify the sensitivity (i.e. derivative of objective function) calculated using the adjoint method with those calculated using finite difference method.

For each objective function, a simple PFLOTRAN simulation is set up, and a unique Python script can be called to compare its sensitivity according to both approaches.

All verifications is stand-alone with regard to the PFLOTRAN simulation. The Python script and especially the adjoint sensitivity calculation uses function in the `src` folder at the root of this repository.

## PFLOTRAN problems

This folder contains default PFLOTRAN simulation of various case to test the different HydrOpTop functions on.

* `quad_128_hetero`: a 2D quad grid (8x16x1) with random permeability in `permeability_field.csv` file.
* `pit_general`: a 2D problem where the domain is splitted in two regions (Pit and Rock). Contains also a `INTEGRAL_FLUX` card that could be use to test connection related functions
* `pit_voronoi`: a 2D problem where the domain is splitted in two regions (Pit and Rock) with the grid in explicit format.
- `2x2x2`: a small problem with 8 elements to test and debug the function.


## Test functions derivative


## Test adjoint derivative

Procedure that compare the total derivative of the cost function compared to finite difference.


## Test filter

A simple PFLOTRAN simulation to verify the head sum objective function (see objective function detail in the `src` folder). Use a cartesian grid with a random permeability field gvien in the `permeability_field.csv` file. Simulations are carried in steady state mode and fully saturated condition.




