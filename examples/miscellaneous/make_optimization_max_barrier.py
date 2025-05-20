"""
Maximize flow in reactive barrier
#################################
"""

import numpy as np
                                  
from HydrOpTop.Functions import Sum_Flux, Volume_Percentage
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Solvers import PFLOTRAN


if __name__ == "__main__":
    pflotranin = "flow_reactive_barrier.in" #path to the PFLOTRAN input file
    sim = PFLOTRAN(
        pflotranin,
        mesh_info='../pflotran_meshes/2D_tri_refined/mesh_info.h5'
    ) #add mesh_info argument to reduce IO

    #get cell ids in the region to optimize (same name than in pflotran input file)
    pit_ids = sim.get_region_ids("Pit") #get the ids corresponding to the excavation
    #create the SIMP parametrization
    perm = Log_SIMP(cell_ids_to_parametrize=pit_ids, property_name="PERMEABILITY", bounds=[5e-12,1e-10], power=3)

    #define cost function
    barrier_connections = sim.get_connections_ids_integral_flux("barrier")
    cf = Sum_Flux(barrier_connections, option="absolute")

    #define volume percentage constrain
    max_vol = (Volume_Percentage(pit_ids), '<', 0.15)

    #define filter to impose a minimum length
    filter = Density_Filter(3)

    #craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], [filter])
    crafted_problem.IO.output_every_iteration(2)
    crafted_problem.IO.output_gradient()
    crafted_problem.IO.define_output_format('vtu')

    p = np.zeros(crafted_problem.get_problem_size(), dtype='f8')+0.145 #initial guess
    crafted_problem.optimize(optimizer="nlopt-mma", action="maximize", max_it=25, initial_guess=p)
  
