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
  parametrization = Identity(all_cells, "a")

  @pytest.mark.parametrize("deriv_mode", [
    pytest.param(["fd",{"scheme":"forward","step":1e-6}], id="fd-forward"),
    pytest.param(["fd",{"scheme":"central","step":1e-6}], id="fd-central"),
    pytest.param(["adjoint", {"method":"direct"}], id="adjoint-direct"),
    pytest.param(["adjoint", {"method":"iterative"}], id="adjoint-iterative")
  ])
  def test_crafter_derivative_level0(self, deriv_mode):
    """
    Test objective function total derivative compared to analytical value with 1 input parametrized
    for both forward and central finite-difference schemes and adjoint
    """
    deriv = deriv_mode[0]
    deriv_args = deriv_mode[1]
    obj = Sum_Variable("x", ids_to_consider=self.all_cells)
    # FD derivative
    craft = Steady_State_Crafter(
        obj, self.solver, [self.parametrization], [], deriv=deriv, deriv_args=deriv_args
    )
    p = np.ones(self.N) * 0.7
    p_bar = craft.pre_evaluation_objective(p)
    cf = craft.evaluate_objective(p_bar)
    J_adjoint = craft.evaluate_total_gradient(obj, p, p_bar=p_bar)

    # Analytical derivative
    J_ana = self.solver.analytical_deriv_dy_dx("a")

    # Comparison
    np.testing.assert_allclose(J_adjoint, J_ana, atol=1e-6, rtol=1e-5)
  
  
  def test_crafter_derivative_level1_diag_filter(self):
    """
    Test adjoint derivative compared to analytical value with a diagonal filter
    """
    h_filter = Heaviside_Filter(self.all_cells)
    obj = Sum_Variable("x", ids_to_consider=self.all_cells)
    # FD derivative
    craft = Steady_State_Crafter(
        obj, self.solver, [self.parametrization], [], [h_filter], deriv="adjoint"
    )
    p = np.random.random(craft.get_problem_size())
    p_bar = craft.pre_evaluation_objective(p)
    cf = craft.evaluate_objective(p_bar)
    J_adjoint = craft.evaluate_total_gradient(obj, p, p_bar=p_bar)

    # Analytical derivative
    J_ana = self.solver.analytical_deriv_dy_dx("a") * h_filter.get_filter_derivative(p)

    # Comparison
    np.testing.assert_allclose(J_adjoint, J_ana, atol=1e-6, rtol=1e-5)


  @pytest.mark.parametrize("filter_", [Density_Filter, Zone_Homogeneous])
  def test_crafter_derivative_level2_general_filter(self, filter_):
    """
    Test adjoint derivative compared to finite difference value for general (non-diagonal) filter
    """
    h_filter = filter_(np.random.choice(self.N, int(self.N)//2, replace=False))

    obj = Sum_Variable("x", ids_to_consider=self.all_cells)
    obj2 = Sum_Variable("x", ids_to_consider=self.all_cells)
    craft = Steady_State_Crafter(
        obj, self.solver, [self.parametrization], [], [h_filter], deriv="adjoint"
    )
    craft_fd = Steady_State_Crafter(
        obj2, self.solver, [self.parametrization], [], [h_filter], deriv="fd", deriv_args={"scheme":"central", "step":1e-6}
    )
    p = np.random.random(craft.get_problem_size())
    p_bar = craft.pre_evaluation_objective(p)
    cf = craft.evaluate_objective(p_bar)
    J_adjoint = craft.evaluate_total_gradient(obj, p, p_bar=p_bar)
    J_fd = craft_fd.evaluate_total_gradient(obj2, p, p_bar=p_bar)

    # Comparison
    np.testing.assert_allclose(J_adjoint, J_fd, atol=1e-6, rtol=1e-5)


  def test_crafter_derivative_level3_filter_sequence(self):
    """
    Test adjoint derivative compared to finite difference value with a set of filter
    """
    from HydrOpTop.Filters import Density_Filter, Zone_Homogeneous, Heaviside_Filter
    obj = Sum_Variable("x", ids_to_consider=self.all_cells)
    # create filter sequence
    reg1 = self.all_cells[:4]
    reg2 = self.all_cells[4:7]
    # this let reg3 with the non filtered cell
    filters = []
    filters.append(Density_Filter(reg1, radius=0.2))
    filters.append(Zone_Homogeneous(reg2))
    filters.append(Heaviside_Filter(reg1))
    # adjoint deriv
    craft = Steady_State_Crafter(obj, self.solver, [self.parametrization], filters=filters)
    p = np.random.random(craft.get_problem_size())
    craft.pre_evaluation_objective(p)
    S_adjoint = craft.evaluate_total_gradient(obj, p)
    # fd deriv
    craft_fd = Steady_State_Crafter(
        copy.deepcopy(obj), self.solver, [self.parametrization], filters=filters, deriv="fd", deriv_args={"scheme":"central","step":1e-6}
    )
    S_fd = craft_fd.evaluate_total_gradient(obj, p)

    assert craft.get_problem_size() == 8 # 4 + 1 + 3
    np.testing.assert_allclose(S_adjoint, S_fd, atol=1e-6, rtol=1e-6)

  @pytest.mark.skip(reason="Not implemented")
  def test_crafter_adjoint_2_param(self): #TODO
    """
    Test adjoint derivative compared to analytical value with 2 input parametrized
    """
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"a")
    solver.create_cell_indexed_dataset(self.b,"b")
    obj = Sum_Variable("x", ids_to_consider=self.all_cells)
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

