import numpy as np
from scipy.linalg import null_space


class Null_Space_Sampler:
    """
    Sample equivalent least-square solutions by perturbing parameters
    in the (near-)null space of the Jacobian, with automatic bound handling.
    """

    def __init__(
        self,
        p,
        jac,
        bound=(0.0, 1.0),
        method="logit",
        rcond=1e-6,
        sigma=0.2,
    ):
        """
        Parameters
        ----------
        p : (n,) ndarray
            Optimal calibrated parameters
        jac : (m, n) ndarray
            Jacobian d(residuals)/dp evaluated at p
        bound : tuple
            (lower, upper) bounds (must be (0,1) for logit)
        method : str
            'logit' (recommended)
        rcond : float
            Relative tolerance for near-null space
        sigma : float
            Std-dev of maximum-entropy perturbation in null space
        """
        self.p = np.asarray(p).copy()
        self.jac_p = np.asarray(jac)
        self.bound = bound
        self.method = method
        self.rcond = rcond
        self.sigma = sigma
        self.reproject = True

        if method != "logit":
            raise NotImplementedError("Only method='logit' is implemented")

        if bound != (0.0, 1.0):
            raise ValueError("Logit transform requires bounds (0,1)")

        # ---- logit transform
        eps = 1e-12
        p_clip = np.clip(self.p, eps, 1.0 - eps)
        self.theta = np.log(p_clip / (1.0 - p_clip))

        # ---- Jacobian in theta space
        D = np.diag(p_clip * (1.0 - p_clip))
        self.jac_theta = self.jac_p @ D

        # ---- null space (near-null allowed)
        self.N = null_space(self.jac_theta, rcond=self.rcond)

        if self.N.size == 0:
            raise ValueError("Jacobian has no (near-)null space")

        self.null_dim = self.N.shape[1]

    def propose_sample(self, size=1):
        """
        Generate new parameter samples.

        Parameters
        ----------
        size : int
            Number of samples

        Returns
        -------
        p_samples : (size, n) ndarray
            New parameter vectors satisfying bounds
        """
        samples = []

        for _ in range(size):
            # maximum-entropy perturbation
            z = np.random.randn(self.null_dim)
            dtheta = self.sigma * (self.N @ z)

            theta_new = self.theta + dtheta
            p_new = 1.0 / (1.0 + np.exp(-theta_new))
            if self.reproject:
                self.__reproject__(p_new)

            samples.append(p_new)

        return np.squeeze(np.array(samples))

    def __reproject__(self, p_new):
        delta = np.linalg.lstsq(
            self.jac_p,
            -self.jac_p @ (p_new - self.p),
            rcond=1e-3,
        )[0]

        p_new += delta
        p_new[:] = np.clip(p_new, 1e-12, 1 - 1e-12)
        return
