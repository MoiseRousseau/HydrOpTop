import numpy as np
import multiprocessing as mp


class Sensitivity_Finite_Difference:
    """
    A simple finite difference sensitivity analysis.
    """
    def __init__(self, eval_obj, current_obj_val=None, scheme="central", step=1e-3, parallel=1):
        #vector
        self.func = eval_obj
        self.scheme = scheme
        self.step = step
        self.parallel = parallel
        self.current_obj_val = current_obj_val
        self.l0 = None
        return
    
    def set_current_obj_val(self, v):
        self.current_obj_val = v
        return

    def compute_sensitivity(self, p):
        """
        Compute the total cost function derivative according to material density
        parameter p.
        
        :param p: the material parameter
        """
        p = p.copy()
        S = np.zeros_like(p)
        for i in range(len(p)):
            if self.scheme == "forward":
                if self.current_obj_val is None:
                    self.current_obj_val = self.func(p)
                p1 = p.copy()
                p1[i] += self.step
                if p1[i] >= 1: 
                    p1[i] = 1.
                    step_ = p1[i] - p[i]
                else:
                    step_ = self.step
                S[i] = (self.func(p1) - self.current_obj_val) / step_
            elif self.scheme == "central":
                # TODO: what if p1 p2 < 0 or > 1 ?
                p1 = p.copy()
                p1[i] += self.step
                p2 = p.copy()
                p2[i] -= self.step
                S[i] = 0.5 * (self.func(p1) - self.func(p2)) / self.step
        return S
