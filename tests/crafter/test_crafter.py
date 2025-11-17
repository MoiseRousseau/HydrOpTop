import numpy as np
import pytest
import copy

from HydrOpTop.Solvers import Dummy_Simulator
from HydrOpTop.Adjoints import Sensitivity_Steady_Simple
from HydrOpTop.Functions import Sum_Variable
from HydrOpTop.Materials import Identity
from HydrOpTop.Crafter import Steady_State_Crafter
from HydrOpTop.Filters import Density_Filter, Heaviside_Filter, Zone_Homogeneous


class Test_Crafter:
    
  # Inputs
  N = 10
  solver = Dummy_Simulator(problem_size=N, seed=1234)
  all_cells = solver.get_region_ids("__all__")
  obj = Sum_Variable("x")
  parametrization = Identity(all_cells, "a")

  @pytest.mark.parametrize("deriv_mode", [
    ["fd",{"scheme":"forward","step":1e-6}],
    ["fd",{"scheme":"central","step":1e-6}],
    ["adjoint", {"method":"direct"}],
    ["adjoint", {"method":"iterative"}]
  ])
  def test_crafter_derivative_1_param(self, deriv_mode):
    """
    Test adjoint derivative compared to analytical value with 1 input parametrized
    for both finite-difference schemes: forward and central.
    The analytical derivative is exact only when not considering filter
    """
    deriv = deriv_mode[0]
    deriv_args = deriv_mode[1]
    # FD derivative
    craft = Steady_State_Crafter(
        self.obj, self.solver, [self.parametrization], [], deriv=deriv, deriv_args=deriv_args
    )
    p = np.ones(self.N) * 0.7
    p_bar = craft.pre_evaluation_objective(p)
    cf = craft.evaluate_objective(p_bar)
    J_adjoint = craft.evaluate_total_gradient(self.obj, p, p_bar=p_bar)

    # Analytical derivative
    J_ana = self.solver.analytical_deriv_dy_dx("a")

    # Comparison
    np.testing.assert_allclose(J_adjoint, J_ana, atol=1e-6, rtol=1e-5)
  
  
  def test_crafter_adjoint_2_param(self): #TODO
    """
    Test adjoint derivative compared to analytical value with 2 input parametrized
    """
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"a")
    solver.create_cell_indexed_dataset(self.b,"b")
    obj = Sum_Variable("x", solved=True)
    parametrizationA = Identity("all", "a")
    parametrizationb = Identity("all", "b")
    solver.run()
    obj.set_inputs([solver.get_output_variable("x")])
    craft = Steady_State_Crafter(obj, solver, [parametrizationA, parametrizationb], [])
    S_adjoint = craft.evaluate_total_gradient(obj, self.b)
    #analytic deriv
    S_ana = solver.analytical_deriv_dy_dx('a') + solver.analytical_deriv_dy_dx('b')
    print(S_ana, S_adjoint)
    assert np.allclose(S_adjoint, S_ana, atol=1e-6, rtol=1e-6) 
  
  
  def test_crafter_adjoint_with_filter_1_param_ana(self):
    """
    Test adjoint derivative compared to analytical value with filter
    """
    h_filter = Heaviside_Filter(self.all_cells)
    # FD derivative
    craft = Steady_State_Crafter(
        self.obj, self.solver, [self.parametrization], [], [h_filter], deriv="adjoint"
    )
    p = np.random.random(craft.get_problem_size())
    p_bar = craft.pre_evaluation_objective(p)
    cf = craft.evaluate_objective(p_bar)
    J_adjoint = craft.evaluate_total_gradient(self.obj, p, p_bar=p_bar)

    # Analytical derivative
    J_ana = self.solver.analytical_deriv_dy_dx("a") * h_filter.get_filter_derivative(p)

    # Comparison
    np.testing.assert_allclose(J_adjoint, J_ana, atol=1e-6, rtol=1e-5)


  @pytest.mark.parametrize("filter_", [Density_Filter, Zone_Homogeneous])
  def test_crafter_adjoint_with_filter_1_param(self, filter_):
    """
    Test adjoint derivative compared to analytical value with filter
    """
    h_filter = filter_(np.random.choice(self.N, int(self.N)//2, replace=False))
    obj2 = copy.deepcopy(self.obj)
    craft = Steady_State_Crafter(
        self.obj, self.solver, [self.parametrization], [], [h_filter], deriv="adjoint"
    )
    craft_fd = Steady_State_Crafter(
        obj2, self.solver, [self.parametrization], [], [h_filter], deriv="fd", deriv_args={"scheme":"central"}
    )
    p = np.random.random(craft.get_problem_size())
    p_bar = craft.pre_evaluation_objective(p)
    cf = craft.evaluate_objective(p_bar)
    J_adjoint = craft.evaluate_total_gradient(self.obj, p, p_bar=p_bar)
    J_fd = craft_fd.evaluate_total_gradient(self.obj, p, p_bar=p_bar)

    # Comparison
    np.testing.assert_allclose(J_adjoint, J_fd, atol=1e-6, rtol=1e-5)


  def test_crafter_adjoint_with_2_filters_1_param_A(self):
    """
    Test adjoint derivative compared to analytical value with two filters
    """
    from HydrOpTop.Filters import Heaviside_Filter
    p = self.A
    solver = Dummy_Simulator(problem_size=len(p))
    obj = Sum_Variable("x", solved=True)
    parametrizationA = Identity("all", "a")
    solver.create_cell_indexed_dataset(self.b,"b")
    filters = [Heaviside_Filter(), Heaviside_Filter()]
    craft = Steady_State_Crafter(obj, solver, [parametrizationA], filters=filters)
    craft.pre_evaluation_objective(p)
    #crafter adjoint
    S_adjoint = craft.evaluate_total_gradient(obj, p)
    #analytic deriv
    A_filtered = filters[1].get_filtered_density(filters[0].get_filtered_density(p))
    solver.create_cell_indexed_dataset(A_filtered,"a")
    solver.run()
    obj.set_inputs([solver.get_output_variable("x")])
    grad = filters[0].get_filter_derivative(p)
    grad *= filters[1].get_filter_derivative(filters[0].get_filtered_density(p))
    S_ana = solver.analytical_deriv_dy_dx('a')*grad
    
    print(S_ana, S_adjoint)
    assert np.allclose(S_adjoint, S_ana, atol=1e-6, rtol=1e-6)
    

