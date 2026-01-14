import numpy as np
import multiprocessing as mp


class Sensitivity_Ensemble:
    """
    Ensemble gradient as in https://arxiv.org/abs/2104.07811
    Not the original paper but a nice and convenient description
    """
    def __init__(self, eval_obj, n_realizations, current_obj_val=None, step=13e-3, parallel=1, **kwargs):
        #vector
        self.func = eval_obj
        self.n_realization = n_realizations
        self.step = step
        self.parallel = parallel
        self.current_obj_val = current_obj_val
        self.l0 = None
        return
    
    def set_current_obj_val(self, v):
        self.current_obj_val = v
        return

    def compute_sensitivity(self, func, filter_sequence, p):
        """
        Compute the total cost function derivative according to material density
        parameter p.
        
        :param p: the material parameter
        """
        p = p.copy()
        N = len(p)
        M = self.n_realization
        feval = 0
        if isinstance(self.current_obj_val,float):
            S = np.zeros_like(p)
        else: #this is an array
            S = np.zeros( (len(p),len(self.current_obj_val)) )
        
        # sample multivariate normal sample around the given p
        rng = np.random.default_rng()
        cov = np.eye(N)*self.step
        p_pop = rng.multivariate_normal(p, cov, M)

        # evaluate function
        f_pop = []
        for i in range(M):
            # For instance with compute derivative as is there is no bounds
            p = p_pop[i]
            rea = self.func(p)
            f_pop.append(rea)
            feval += 1
        
        # estimate covariance
        Dp = p_pop - np.mean(p_pop, axis=0)
        Df = f_pop - np.mean(f_pop, axis=0)
        Cxz = 1/M * Dp.T @ Df
        S = Cxz # as cov is always diagonal (S = cov^-1 @ Cxz)

        return S, feval
