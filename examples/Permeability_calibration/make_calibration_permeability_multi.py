"""
Multi-material permeability field calibration
#############################################

Example of the calibration of two materials defined in PFLOTRAN script.
TODO
"""

import numpy as np
                                  
from HydrOpTop.Functions import Reference_Liquid_Head #objective
from HydrOpTop.Functions import Volume_Percentage #constrain
from HydrOpTop.Materials import Log_SIMP, MultiMaterials
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
    sand_cells = sim.get_region_ids("sand")

    perm_till = Log_SIMP(
        cell_ids_to_parametrize=till_cells,
        property_name="PERMEABILITY",
        bounds=[1e-14, 2e-13],
        power=1
    )
    perm_sand = Log_SIMP(
        cell_ids_to_parametrize=sand_cells,
        property_name="PERMEABILITY",
        bounds=[1e-13, 1e-11],
        power=1
    )
    perm = MultiMaterials([perm_till, perm_sand])

    #define cost function as sum of the head in the pit
    cell_ids = [444, 789, 920, 1030, 1339]
    head = [230, 250, 227, 146, 210]
    cf = Reference_Liquid_Head(head, cell_ids, norm=2)

    #define maximum volume constrains
    #max_vol = Volume_Percentage("parametrized_cell")
    #max_vol.constraint_tol = 0.2

    filter_ = Density_Filter(10.)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(cf, sim, [perm], [], [filter_])
    crafted_problem.IO.output_every_iteration(10)
    crafted_problem.IO.output_gradient()
    crafted_problem.IO.output_material_properties()
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size()) + 0.5
    p_opt = crafted_problem.optimize(
        optimizer="nlopt-mma",
        action="minimize",
        max_it=100,
        ftol=1e-40,
        initial_guess=p
    )
    
