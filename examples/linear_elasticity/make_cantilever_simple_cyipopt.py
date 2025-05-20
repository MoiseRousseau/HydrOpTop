"""
Using cyipopt instead of mma as optimizer
#########################################
"""

import numpy as np
import time
                                  
from HydrOpTop.Functions import Mechanical_Compliance, Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from linear_elasticity_2d import Linear_Elasticity_2D


if __name__ == "__main__":
    algo = "cyipopt"
    #Create a timer to time the optimization
    t = time.time()

    #Create the simulation object with the linear elasticity solver shield
    sim = Linear_Elasticity_2D("cantilever")
    young_modulus = SIMP(cell_ids_to_parametrize="all", property_name="YOUNG_MODULUS", bounds=[0, 2000], power=3)
    cf = Mechanical_Compliance(ids_to_consider="everywhere")
    max_vol = (Volume_Percentage("parametrized_cell"), '<', 0.5)

    dfilter = Density_Filter(0.3)

    crafted_problem = Steady_State_Crafter(objective=cf, 
                                         solver=sim, 
                                         mat_props=[young_modulus], 
                                         constraints=[max_vol], 
                                         filters=[dfilter])
    crafted_problem.IO.define_output_format("vtu")
    #crafted_problem.IO.no_output_initial()
    crafted_problem.IO.output_every_iteration(2)
    crafted_problem.IO.define_output_log(algo)

    p_ini = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2
    out = crafted_problem.optimize(optimizer=algo, action="minimize", ftol=1e-6, max_it=100, initial_guess=p_ini)

    #Output the final optimized and filtered density parameter in a out.vtu file
    crafted_problem.IO.write_fields_to_file([out.p_opt, out.p_opt_filtered], "./out.vtu", ["Raw_density", "Filtered_density"])

    print(f"Elapsed time: {time.time()-t} seconds")
  
