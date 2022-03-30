import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np
from HydrOpTop.Solvers import Dummy_Simulator
from HydrOpTop.Adjoints import Sensitivity_Steady_Simple
from HydrOpTop.Functions import Sum_Variable
from HydrOpTop.Materials import Identity
from HydrOpTop.Crafter import Steady_State_Crafter


class Test_Crafter:

  rgn = np.random.default_rng(232)
  n = 10
  A = rgn.random(n)
  b = rgn.random(n)


  def test_sensitivity_manual_1_param_A(self):
    """
    Test adjoint derivative compared to analytical value with 1 input parametrized
    """
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"a")
    solver.create_cell_indexed_dataset(self.b,"b")
    obj = Sum_Variable("x", solved=True)
    parametrization = Identity("all", "a")
    solver.run()
    obj.set_inputs([solver.get_output_variable("x")])
    sens = Sensitivity_Steady_Simple("x",
                   [parametrization], solver, np.arange(1,self.n+1)) 
    #compute
    S_adjoint = sens.compute_sensitivity(self.b, 
                                         obj.d_objective_dY(None),
                                         obj.d_objective_dX(None),
                                         []) #p, dc_dYi, dc_dXi, Xi_name):
    #analytic deriv
    S_ana = solver.analytical_deriv_dy_dx('a')
    print(S_ana, S_adjoint)
    assert np.allclose(S_adjoint, S_ana, atol=1e-6, rtol=1e-6)  
  
  def test_crafter_derivative_1_param_A(self):
    """
    Test adjoint derivative compared to analytical value with 1 input parametrized
    """
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"a")
    solver.create_cell_indexed_dataset(self.b,"b")
    obj = Sum_Variable("x", solved=True)
    parametrization = Identity("all", "a")
    solver.run()
    obj.set_inputs([solver.get_output_variable("x")])
    craft = Steady_State_Crafter(obj, solver, [parametrization], [])
    
    S_adjoint = craft.evaluate_total_gradient(obj, self.b)
    #analytic deriv
    S_ana = solver.analytical_deriv_dy_dx('a')
    print(S_ana, S_adjoint)
    assert np.allclose(S_adjoint, S_ana, atol=1e-6, rtol=1e-6)


  def test_sensitivity_manual_1_param_b(self):
    """
    Test adjoint derivative compared to analytical value with 1 input parametrized
    """
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"a")
    solver.create_cell_indexed_dataset(self.b,"b")
    obj = Sum_Variable("x", solved=True)
    parametrization = Identity("all", "b")
    solver.run()
    obj.set_inputs([solver.get_output_variable("x")])
    sens = Sensitivity_Steady_Simple("x",
                   [parametrization], solver, np.arange(1,self.n+1)) 
    #compute
    S_adjoint = sens.compute_sensitivity(self.b, 
                                         obj.d_objective_dY(None),
                                         obj.d_objective_dX(None),
                                         []) #p, dc_dYi, dc_dXi, Xi_name):
    #analytic deriv
    S_ana = solver.analytical_deriv_dy_dx('b')
    print(S_ana, S_adjoint)
    assert np.allclose(S_adjoint, S_ana, atol=1e-6, rtol=1e-6)    
  
  def test_crafter_derivative_1_param_b(self):
    """
    Test adjoint derivative compared to analytical value with 1 input parametrized
    """
    solver = Dummy_Simulator()
    solver.create_cell_indexed_dataset(self.A,"a")
    solver.create_cell_indexed_dataset(self.b,"b")
    obj = Sum_Variable("x", solved=True)
    parametrization = Identity("all", "b")
    solver.run()
    obj.set_inputs([solver.get_output_variable("x")])
    craft = Steady_State_Crafter(obj, solver, [parametrization], [])
    S_adjoint = craft.evaluate_total_gradient(obj, self.b)
    #analytic deriv
    S_ana = solver.analytical_deriv_dy_dx('b')
    print(S_ana, S_adjoint)
    assert np.allclose(S_adjoint, S_ana, atol=1e-6, rtol=1e-6)
  
  
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
  
  
  def test_crafter_adjoint_with_filter_1_param_A(self):
    """
    Test adjoint derivative compared to analytical value with filter
    """
    from HydrOpTop.Filters import Heaviside_Filter
    p = self.A
    solver = Dummy_Simulator(problem_size=len(p))
    obj = Sum_Variable("x", solved=True)
    parametrizationA = Identity("all", "a")
    solver.create_cell_indexed_dataset(self.b,"b")
    filter_ = Heaviside_Filter()
    craft = Steady_State_Crafter(obj, solver, [parametrizationA], filters=[filter_])
    craft.pre_evaluation_objective(p)
    #crafter adjoint
    S_adjoint = craft.evaluate_total_gradient(obj, p)
    #analytic deriv
    A_filtered = filter_.get_filtered_density(p)
    solver.create_cell_indexed_dataset(A_filtered,"a")
    solver.run()
    obj.set_inputs([solver.get_output_variable("x")])
    S_ana = solver.analytical_deriv_dy_dx('a')*filter_.get_filter_derivative(p)
    
    print(S_ana, S_adjoint)
    assert np.allclose(S_adjoint, S_ana, atol=1e-6, rtol=1e-6)
  
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
    

