"""
Simple permeability field calibration
#####################################
"""

import numpy as np
                                  
from HydrOpTop.Functions import Reference_Liquid_Head #objective
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Solvers import PFLOTRAN

if __name__ == "__main__":
    #create PFLOTRAN simulation object
    pflotranin = "pflotran_small.in"
    sim = PFLOTRAN(pflotranin)
    all_cells = sim.get_region_ids("__all__")

    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    perm = Log_SIMP(cell_ids_to_parametrize=all_cells, property_name="PERMEABILITY", bounds=[1e-14, 1e-12], power=1)

    #define cost function as sum of the head in the pit
    cell_ids = [444, 789, 920, 1030, 1339]
    head = [230., 250., 227., 146., 210.]
    cf = Reference_Liquid_Head(head, cell_ids, norm=2)

    filter_ = Density_Filter(cell_ids=all_cells, radius=10.)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(cf, sim, [perm], [], [filter_], deriv="adjoint", deriv_args={"method":"iterative"})
    crafted_problem.IO.output_every_iteration(10)
    crafted_problem.IO.output_gradient()
    crafted_problem.IO.output_material_properties()
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size()) + 0.5
    #p = 0.999*np.random.rand(crafted_problem.get_problem_size())+0.001 
    p_opt = crafted_problem.optimize(
        optimizer="nlopt-ccsaq", action="minimize",
        stop={'ftol_rel':1e-6,"max_it":100},
        initial_guess=p
    )

    crafted_problem.IO.plot_convergence_history()
    
