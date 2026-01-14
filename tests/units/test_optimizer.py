import pytest
import numpy as np
from derivatives.utils import finite_difference_dp


from HydrOpTop.Solvers import Dummy_Simulator
from HydrOpTop.Filters import Density_Filter
from HydrOpTop.Materials import SIMP
from HydrOpTop.Functions import Sum_Variable, Volume_Percentage
from HydrOpTop.Crafter import Steady_State_Crafter


class Test_Optimizer:
    # Inputs
    N = 10
    solver = Dummy_Simulator(problem_size=N, seed=1234)
    all_cells = solver.get_region_ids("__all__")
    param = SIMP(all_cells, "a", [0.,1.])
    filter = Density_Filter(all_cells, 1.)
    obj = Sum_Variable("x", ids_to_consider=solver.get_region_ids("__all__"))
    constraint_l = (Volume_Percentage(all_cells), ">", 0.1)
    constraint_s = (Volume_Percentage(all_cells), "<", 0.3)
    craft = Steady_State_Crafter(
        obj, solver, [param], filters=[filter],
        constraints=[constraint_l, constraint_s],
        deriv="adjoint", deriv_args={"method":"direct"}
    )
    p = np.random.random(craft.get_problem_size())


    @pytest.mark.parametrize("method", [
        pytest.param("scipy_function_to_optimize", id="scipy"),
        pytest.param("nlopt_function_to_optimize", id="nlopt"),
    ])
    def test_optimizer_value(self, method):
        """
        Check if all optimizer wrapper return the same objective function
        """
        func = getattr(self.craft, method)
        ref_val = self.craft.nlopt_function_to_optimize(self.p) #TODO change to analytical value
        assert func(self.p) == ref_val
        return


    @pytest.mark.parametrize("method", [
        pytest.param("scipy_jac", id="scipy"),
    ])
    def test_optimizer_derivative(self, method):
        """
        Test if all optimizer have same derivative
        """
        dobj_dp_ref = finite_difference_dp(
            lambda x: self.craft.evaluate_objective(self.craft.pre_evaluation_objective(x)),
            self.p, eps=1e-4,
        )
        func = getattr(self.craft, method)
        grad = func(self.p)
        np.testing.assert_allclose(grad, dobj_dp_ref, atol=1e-6, rtol=1e-5)
        return

    @pytest.mark.parametrize("method", [
        pytest.param("nlopt_function_to_optimize", id="nlopt"),
    ])
    def test_optimizer_derivative_inplace(self, method):
        """
        Test if all optimizer have same derivative (in place gradient version)
        """
        dobj_dp_ref = finite_difference_dp(
            lambda x: self.craft.evaluate_objective(self.craft.pre_evaluation_objective(x)),
            self.p, eps=1e-4,
        )
        grad = np.zeros_like(dobj_dp_ref)
        func = getattr(self.craft, method)
        func(self.p, grad)
        np.testing.assert_allclose(grad, dobj_dp_ref, atol=1e-6, rtol=1e-5)
        return


    @pytest.mark.parametrize("methods", [
        pytest.param(("scipy_constraint_val","scipy_constraint_jac",0), id="scipy_inf"),
        pytest.param(("scipy_constraint_val","scipy_constraint_jac",1), id="scipy_sup"),
    ])
    def test_optimizer_dconstraint(self, methods):
        """
        Test if all optimizer have same derivative
        """
        func_str, dfunc_str, iconst = methods
        func = getattr(self.craft, func_str)
        dfunc = getattr(self.craft, dfunc_str)
        const = self.craft.constraints[iconst]
        dc_dp_ref = finite_difference_dp(
            lambda x: func(const, x), self.p, eps=1e-4,
        )
        dc_dp = dfunc(const, self.p)
        np.testing.assert_allclose(dc_dp, dc_dp_ref, atol=1e-6, rtol=1e-5)
        return
    
    @pytest.mark.parametrize("methods", [
        pytest.param(("__nlopt_generic_constraint_to_optimize__",0), id="nlopt_inf"),
        pytest.param(("__nlopt_generic_constraint_to_optimize__",1), id="nlopt_sup"),
    ])
    def test_optimizer_dconstraint_inplace(self, methods):
        """
        Test if all optimizer have same derivative
        """
        func_str, iconst = methods
        func = getattr(self.craft, func_str)
        dc_dp_ref = finite_difference_dp(
            lambda x: func(x, iconstraint=iconst), self.p, eps=1e-4,
        )
        grad = np.zeros_like(dc_dp_ref)
        dc_dp = func(self.p, grad, iconst)
        np.testing.assert_allclose(grad, dc_dp_ref, atol=1e-6, rtol=1e-5)
        return