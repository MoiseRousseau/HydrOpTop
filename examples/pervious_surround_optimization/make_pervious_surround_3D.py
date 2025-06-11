"""
3D pervious surround
====================

Source file for optimizing a man-made pervious surrond to reduce advective flow through a backfilled pit as in Rousseau and Pabst (2021) (https://doi.org/10.1007/s00158-022-03266-1)
"""

import numpy as np
import time

from HydrOpTop.Functions import p_Weighted_Head_Gradient, Volume_Percentage
from HydrOpTop.Materials import Log_SIMP
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Solvers import PFLOTRAN


if __name__ == "__main__":
    #1. Create a timer to time the optimization
    t = time.time()

    #2. create PFLOTRAN simulation object
    pflotranin = "pflotran_perv_surr_3D.in"
    sim = PFLOTRAN(pflotranin)

    #3. Parametrize hydraulic conductivity
    #get cell ids in the region to optimize and parametrize permeability
    #same name than in pflotran input file
    pit_ids = sim.get_region_ids("Pit")
    perm = Log_SIMP(cell_ids_to_parametrize=pit_ids, property_name="PERMEABILITY", bounds=[5e-14,1e-10], power=3)

    #4. Define the cost function and constraint
    #define cost function as sum of the head in the pit
    cf = p_Weighted_Head_Gradient(pit_ids, invert_weighting=True)
    #define maximum volume constrains
    max_vol = (Volume_Percentage(pit_ids), '<', 0.2)

    #5. Define filter
    filter = Density_Filter(6) #search neighbors in a 6 m radius

    #6. Craft optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(cf, sim, [perm], [max_vol], [filter])

    #7. Output
    #Define the output behavior (output density parameter every 5 iterations in vtu format)
    crafted_problem.IO.output_every_iteration(5)
    crafted_problem.IO.define_output_format("vtu")

    #8. Degine initial guess (homogeneous) and optimize
    p_ini = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.12
    out = crafted_problem.optimize(optimizer="nlopt-mma", action="minimize", max_it=200, ftol=1e-8, initial_guess=p_ini)

    #9. Output the final optimized and filtered density parameter in a out.vtu file
    crafted_problem.IO.write_fields_to_file([out.p_opt_filtered], "./out.vtu", ["Filtered_density"], var_loc="cell", at_ids=pit_ids-1)
    print(f"Elapsed time: {time.time()-t} seconds")
    crafted_problem.IO.plot_convergence_history()
    
    print("Elapsed time:", time.time()-t, "s")
  
