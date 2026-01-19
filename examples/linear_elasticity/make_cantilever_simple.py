"""
Verification against classical cantilever benchmark
===================================================

Problem description
-------------------

We used HydrOpTop to solve a the cantilever classical design optimisation benchmark in linear mechanic to validate the implementation.
The design domain is a rectangle of 5 units high and 10 units wide discretized with triangles, where the cantilever is fixed at the left boundary and the load (unitary) applied at right bottom corner.
The 2D linear elasticity solver is used. 
The design domain is discretized in 11856 triangles and 6079 nodes generated using `Salome <www.salome-platform.org>`_ CAD software.
Mesh is then exported and BC and loads are defined in the format required by the solver (see solver documentation `here <https://github.com/MoiseRousseau/MinimalFem-For-Topology-Optimization>`_)

The topology optimization is carried out by considering a parametrization of the Young modulus in each mesh triangle using a SIMP parametrization with a penalization power of 3.
A ball density filter with a radius of  is also applied to avoid the formation of checkerboard pattern and to apply a minimal size constraint of 0.3 units.
A volume constraint of 50% is considered.

The Python Code
---------------
"""

import numpy as np

from mechanical_compliance import Mechanical_Compliance
from HydrOpTop.Functions import Volume_Percentage
from HydrOpTop.Materials import SIMP
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Crafter import Steady_State_Crafter
from linear_elasticity_2d import Linear_Elasticity_2D

if __name__ == "__main__":
    #Create the simulation object with the linear elasticity solver shield
    sim = Linear_Elasticity_2D("cantilever")
    all_cells = sim.get_region_ids("__all__")
    all_nodes = sim.get_node_region_ids("__all__")

    #Parametrize the cell of the whole mesh using a SIMP parametrization of the Young modulus between 0 (p=0) and 2000 MPa (p=1)
    young_modulus = SIMP(
        cell_ids_to_parametrize=all_cells,
        property_name="YOUNG_MODULUS",
        bounds=[0, 2000], power=3
    )

    #Define the cost function (mechanical compliance) defined at all the nodes of the simulation
    cf = Mechanical_Compliance(ids_to_consider=all_nodes)

    #Define the maximum volume constrain
    max_vol = (Volume_Percentage(all_cells), '<', 0.5)

    #Create the density filter using a ball radius of 0.3 units to avoid checkerboard effect
    dfilter = Density_Filter(all_cells, radius=0.3)

    #Craft the optimization problem
    #i.e. create function to optimize, initiate IO array in classes...
    crafted_problem = Steady_State_Crafter(
        objective=cf, solver=sim, mat_props=[young_modulus],
        constraints=[max_vol], filters=[dfilter],
        deriv="adjoint",deriv_args={"method":"direct"},
    )

    #Define the output behavior (output density parameter every 2 iterations in vtu format)
    crafted_problem.IO.output_every_iteration(2)
    crafted_problem.IO.define_output_format("vtu")
    crafted_problem.IO.output_gradient()
    crafted_problem.IO.output_gradient_constraints()

    #Create a initial guess for the optimization
    p_ini = np.zeros(crafted_problem.get_problem_size(),dtype='f8') + 0.2

    #Use the MMA algorithm from the library ``nlopt`` to minimize the cost function, until a maximum of 50 iterations is reached or when the relative variation of the cost function is below 0.0001
    out = crafted_problem.optimize(
        optimizer="nlopt-ccsaq", action="minimize", initial_guess=p_ini,
        optimizer_args={"ftol":0.0001, "set_maxeval":50, "set_ftol_rel":1e-3}
    )

    #Output the final optimized and filtered density parameter in a out.vtu file
    crafted_problem.IO.write_fields_to_file([out.p_opt_filtered], "./out.vtu", ["Filtered_density"])



# %%
#Results
#-------
# 
#.. figure:: ./results.gif
#   :alt: Optimization results
#

