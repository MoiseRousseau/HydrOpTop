# PFLOTRAN topology optimization verification

This folder aimed to verify the sensitivity (i.e. derivative of objective function) calculated using the adjoint method with those calculated using finite difference method.

For each objective function, a simple PFLOTRAN simulation is set up, and a unique Python script can be called to compare its sensitivity according to both approaches.

All verifications is stand-alone with regard to the PFLOTRAN simulation. The Python script and especially the adjoint sensitivity calculation uses function in the `src` folder at the root of this repository.

## Sum head objective

A simple PFLOTRAN simulation to verify the head sum objective function (see objective function detail in the `src` folder). Use a cartesian grid with a random permeability field gvien in the `permeability_field.csv` file. Simulations are carried in steady state mode and fully saturated condition.



## Flux through a integral face


