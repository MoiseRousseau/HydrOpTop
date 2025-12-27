"""
Free field permeability calibration
-----------------------------------


Example of the calibration of two materials of a PFLOTRAN simulation.
This version use a free field parametrization with a gaussian correlation using a density filter
"""

import numpy as np
                                  
from HydrOpTop.Functions import Least_Square_Calibration
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Solvers import PFLOTRAN

if __name__ == "__main__":
    #create PFLOTRAN simulation object
    pflotranin = "./pflotran_multi.in"
    sim = PFLOTRAN(pflotranin)

    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    till_cells = sim.get_region_ids("till")
    perm_till = Log_SIMP(
        cell_ids_to_parametrize=till_cells,
        property_name="PERMEABILITY",
        bounds=[1e-14, 2e-13],
        power=1
    )
    filter_till = Density_Filter(till_cells, radius=20.)

    sand_cells = sim.get_region_ids("sand")
    perm_sand = Log_SIMP(
        cell_ids_to_parametrize=sand_cells,
        property_name="PERMEABILITY",
        bounds=[1e-13, 1e-11],
        power=1
    )
    filter_sand = Density_Filter(sand_cells, radius=20.)

    #define cost function as sum of the head in the pit
    cell_ids = [444, 789, 920, 1030, 1339]
    head = [230, 250, 227, 146, 210]
    cf = Least_Square_Calibration(head, cell_ids)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(
        objective=cf, solver=sim,
        mat_props=[perm_till, perm_sand],
        filters=[filter_till, filter_sand],
        deriv="adjoint",
        deriv_args={"method":"direct"}
    )
    crafted_problem.IO.output_every_iteration(1)
    crafted_problem.IO.output_gradient()
    crafted_problem.IO.output_material_properties()
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size()) + 0.5
    p_opt = crafted_problem.optimize(
        optimizer="scipy-trf",
        action="minimize",
        max_it=10,
        stop={'ftol':1e-40, "xtol":1e-4},
        initial_guess=p
    )

    crafted_problem.IO.plot_convergence_history()
    
