import numpy as np
from .sensitivity_steady_simple import Sensitivity_Steady_Simple
from scipy.linalg import solve

DEFAULT_SOLVER_ARGS = {
    "step":1e-3,
    "correction":1,
}

class Sensitivity_Steady_Adjoint_Corrected:
    def __init__(self,
            solved_vars, parametrized_mat_props, solver, p_ids, func, solver_args={},
        ):
        self.sensitivity_adjoint = Sensitivity_Steady_Simple(
            solved_vars, parametrized_mat_props, solver, p_ids, solver_args,
        )
        solver_args_ = DEFAULT_SOLVER_ARGS.copy()
        solver_args_.update(solver_args)
        self.func = func
        self.step = solver_args_.pop("step")
        self.correction = solver_args_.pop("correction")
        self.current_obj_val = np.nan
        return


    def set_current_obj_val(self, v):
        self.current_obj_val = v
        return


    def compute_sensitivity(self, func, filter_sequence, p):
        # Base adjoint
        Japprox = self.sensitivity_adjoint.compute_sensitivity(func, filter_sequence, p)
        self.l0 = self.sensitivity_adjoint.l0

        # Sample correction
        p_ = p.copy()
        Dp = []
        df = []
        if self.current_obj_val is None:
            self.current_obj_val = self.func(p)
        for i in range(self.correction):
            p1 = p_ + (np.random.random(len(p)) - 0.5) * self.step
            df.append(self.func(p1) - self.current_obj_val)
            Dp.append(p1-p_)
        p[:] = p_

        # Compute correction
        U = np.column_stack(Dp)  # n × k
        V = np.column_stack(df)  # m × k

        J = self.__minimal_angle_update__(U, V, Japprox)

        return J, self.correction

    def __minimal_angle_update__(self, U, V, Ja):
        """
        Solve (Ja + DJ)^T U = V
        with minimal angle between J and Ja
        (i.e. minimal Frobenius norm of DJ)
        """
        R = V - Ja.T @ U              # (m, k)
        U_pinv = np.linalg.pinv(U)    # (k, n)
        DJ = U_pinv.T @ R.T           # (n, m)
        return Ja + DJ

        