"""
2D permeability field calibration using Pilot Points
####################################################
"""

import numpy as np

from HydrOpTop.Solvers import PFLOTRAN
from HydrOpTop.Functions import Reference_Liquid_Head #objective
from HydrOpTop.Functions import Volume_Percentage #constrain
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Pilot_Points

if __name__ == "__main__":

    #create PFLOTRAN simulation object
    pflotranin = "pflotran_pp.in"
    sim = PFLOTRAN(pflotranin)

    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    perm = Log_SIMP(cell_ids_to_parametrize="all", property_name="PERMEABILITY", bounds=[1e-14, 1e-12], power=1)

    #define cost function as sum of the head in the pit
    cell_ids = [444, 789, 920, 1030, 1339]
    head = [230, 250, 227, 146, 210]
    cf = Reference_Liquid_Head(head, cell_ids, norm=2)

    # Parametrize with Pilot Points
    rng = np.random.default_rng(seed=1234) # for reproductibility
    control_points = np.random.rand(30,2)*200-100
    ppfilter = Pilot_Points(
        control_points,
        parametrized_cells="all",
        interpolator="RBFInterpolator",
    )

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(cf, sim, [perm], [], [ppfilter])
    crafted_problem.IO.output_every_iteration(10)
    crafted_problem.IO.output_gradient()
    crafted_problem.IO.output_material_properties()
    crafted_problem.IO.define_output_format("vtu")

    #initialize optimizer
    p = np.zeros(crafted_problem.get_problem_size()) + 0.5
    print(p.shape)
    #p = 0.999*np.random.rand(crafted_problem.get_problem_size())+0.001 
    p_opt = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=100, ftol=0.0001, initial_guess=p)
    
