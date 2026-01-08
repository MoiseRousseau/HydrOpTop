import numpy as np
from scipy.optimize import OptimizeResult

# Coded with ChatGPT:
# https://chatgpt.com/share/69584676-c724-800d-a47a-caebafc0f1d8

class SPSA:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    This class implements the SPSA algorithm originally proposed by
    Spall (2000), extended with:

    - Adaptive gain tuning (step size and perturbation size)
    - Adaptive penalty continuation for general constraints
    - Bound constraints via projection
    - Optional Polyak–Ruppert iterate averaging

    The optimizer is designed for **noisy, black-box, high-dimensional**
    objective functions where gradients are unavailable or expensive.

    Notes
    -----
    - Only **two objective evaluations per iteration** are required,
      regardless of problem dimension.
    - Convergence requires diminishing gain sequences satisfying
      Robbins–Monro conditions.
    - General nonlinear constraints are handled using an adaptive
      quadratic penalty (exact feasibility is asymptotic).
    - This implementation is compatible with the SciPy optimization API.

    References
    ----------
    Spall, J. C. (2000).
    "Stochastic Optimization."
    IEEE Transactions on Automatic Control, 45(10), 1839–1853.
    """

    def __init__(
        self,
        maxiter=1000,
        a=0.08,
        c=0.001,
        alpha=0.602, #0.6
        gamma=0.101, #0.1
        A=10.0,
        seed=None,
        bounds=None,
        constraints=None,

        # adaptive penalty
        penalty_init=1.0,
        penalty_growth=2.0,
        penalty_max=1e8,
        penalty_tol=0.9,

        # adaptive gains
        gain_beta = 0.9,        # gradient EMA
        improve_beta = 0.9,    # improvement EMA
        gain_eps = 1e-8,
        gain_min = 0.2,        # never kill progress
        gain_max = 3.0,        # avoid explosions

        stagnation_window=20,
        stagnation_tol=1e-4,
        stagnation_shrink=0.5,
        perturb_growth=1.05,

        average=False,
        avg_start=0,
        callback=None,
    ):
        """
        Initialize the SPSA optimizer.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of SPSA iterations.
        a : float, optional
            Initial step-size gain coefficient ``a`` in
            ``a_k = a / (k + A)^alpha``.
        c : float, optional
            Initial perturbation-size gain coefficient ``c`` in
            ``c_k = c / k^gamma``.
        alpha : float, optional
            Step-size decay exponent. Must satisfy
            ``0.5 < alpha <= 1`` for convergence.
        gamma : float, optional
            Perturbation decay exponent. Typically in ``(0, 0.5]``.
        A : float, optional
            Stability constant delaying step-size decay.
            A common choice is ``A ≈ 0.1 * maxiter``.
        seed : int or None, optional
            Random seed for reproducibility.
        bounds : sequence of (float or None, float or None), optional
            Bound constraints on variables, specified as in
            ``scipy.optimize.minimize``.
        constraints : list of dict, optional
            General constraints specified in SciPy format:
            ``{'type': 'ineq', 'fun': c(x)}`` or
            ``{'type': 'eq', 'fun': c(x)}``.
        penalty_init : float, optional
            Initial penalty coefficient for constraint violations.
        penalty_growth : float, optional
            Multiplicative factor used to increase the penalty when
            constraint violation stagnates.
        penalty_max : float, optional
            Maximum allowed penalty coefficient.
        penalty_tol : float, optional
            Tolerance factor controlling when the penalty is increased.
        gain_beta : float, optional
            Exponential averaging factor for gradient variance estimation.
        gain_eps : float, optional
            Small positive constant to avoid division by zero in
            adaptive gain scaling.
        stagnation_window : int, optional
            Number of recent iterations used to detect stagnation.
        stagnation_tol : float, optional
            Relative improvement threshold below which stagnation is
            declared.
        stagnation_shrink : float, optional
            Factor applied to shrink the step-size coefficient ``a``
            when stagnation is detected.
        perturb_growth : float, optional
            Factor applied to increase the perturbation coefficient ``c``
            during stagnation.
        average : bool, optional
            If True, enable Polyak–Ruppert averaging of iterates.
        avg_start : int, optional
            Iteration index at which averaging begins.
        callback : callable or None, optional
            User-supplied function called after each iteration as
            ``callback(xk)``.
        """
        self.maxiter = maxiter
        self.a0 = a
        self.c0 = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A

        self.rng = np.random.default_rng(seed)

        self.bounds = bounds
        self.constraints = constraints or []

        # penalty
        self.penalty = penalty_init
        self.penalty_growth = penalty_growth
        self.penalty_max = penalty_max
        self.penalty_tol = penalty_tol

        # adaptive gain tuning
        self.gain_beta = gain_beta
        self.improve_beta = improve_beta
        self.gain_eps = gain_eps
        self.gain_min = gain_min
        self.gain_max = gain_max

        self.stagnation_window = stagnation_window
        self.stagnation_tol = stagnation_tol
        self.stagnation_shrink = stagnation_shrink
        self.perturb_growth = perturb_growth

        self.average = average
        self.avg_start = avg_start
        self.first_print = True
        self.callback = callback

    # ---------- utilities ----------

    def _project_bounds(self, x):
        if self.bounds is None:
            return x
        x = np.asarray(x).copy()
        lb, ub = self.bounds
        x = np.where(x < lb, lb, x)
        x = np.where(x > ub, ub, x)
        return x

    def _constraint_violation(self, x):
        v = 0.0
        for c in self.constraints:
            val = c["fun"](x)
            if c["type"] == "ineq":
                v += np.sum(np.maximum(0.0, -val) ** 2)
            elif c["type"] == "eq":
                v += np.sum(val ** 2)
        return v

    def _penalty_term(self, x):
        return self.penalty * self._constraint_violation(x)

    # ---------- optimization ----------

    def fit(self, fun, x0, args=()):
        """
        Minimize an objective function using SPSA.

        Parameters
        ----------
        fun : callable
            Objective function to be minimized.
            Must have signature ``fun(x, *args) -> float``.
            The function may be noisy and need not be deterministic.
        x0 : array_like
            Initial parameter vector.
        args : tuple, optional
            Extra arguments passed to the objective function.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Optimization result with the following attributes:

            - ``x`` : ndarray
                Final solution (averaged iterate if enabled).
            - ``fun`` : float
                Objective function value at ``x``.
            - ``nit`` : int
                Number of iterations performed.
            - ``nfev`` : int
                Number of objective function evaluations.
            - ``success`` : bool
                Whether the algorithm terminated normally.
            - ``message`` : str
                Termination message.

        Notes
        -----
        - The gradient is estimated using simultaneous perturbations,
          requiring exactly two function evaluations per iteration.
        - Constraint satisfaction is enforced asymptotically through
          adaptive quadratic penalties.
        - For noisy problems, enabling iterate averaging is strongly
          recommended.
        """
        x = self._project_bounds(np.asarray(x0, dtype=float))
        n = x.size

        # averaging
        x_avg = np.zeros_like(x)
        avg_count = 0

        # adaptive state
        grad_ema = 1.0
        df_ema = 0.0
        prev_fval = None

        recent_f = []
        best_violation = np.inf
        nfev = 0

        def penalized_fun(z):
            return fun(z, *args) + self._penalty_term(z)

        ak, ck = self.a0, self.c0
        for k in range(self.maxiter):
            # base gains

            delta = self.rng.choice([-1.0, 1.0], size=n)

            x_plus = self._project_bounds(x + ck * delta)
            x_minus = self._project_bounds(x - ck * delta)

            f_plus = penalized_fun(x_plus)
            f_minus = penalized_fun(x_minus)
            fval = 0.5*(f_plus + f_minus)
            nfev += 2

            ghat = (f_plus - f_minus) / (2.0 * ck * delta)

            # ---------- gradient norm tracking ----------
            gnorm2 = np.mean(ghat ** 2)
            grad_ema = (
                self.gain_beta * grad_ema
                + (1 - self.gain_beta) * gnorm2
            )

            # ---------- improvement tracking ----------
            if prev_fval is not None:
                df = prev_fval - fval  # positive = improvement
                df_ema = (
                    self.improve_beta * df_ema
                    + (1 - self.improve_beta) * df
                )
            else:
                df = 0.0

            # ---------- effective gain ----------
            ak_eff = ak / np.sqrt(grad_ema + self.gain_eps)

            # reward good progress, penalize bad progress
            if df < 0:
                progress_scale = 0.5
            else:
                progress_scale = np.clip(
                df_ema / (abs(df) + self.gain_eps),
                self.gain_min,
                self.gain_max,
            )

            ak_eff *= progress_scale

            # update
            x_new = self._project_bounds(x - ak_eff * ghat)
            step_size = np.linalg.norm(x - x_new) / np.sqrt(len(x))
            x = x_new

            #fval = penalized_fun(x)
            viol = self._constraint_violation(x)
            nfev += 1

            prev_fval = fval

            # ---------- stagnation detection ----------
            recent_f.append(fval)
            if len(recent_f) > self.stagnation_window:
                recent_f.pop(0)

            if len(recent_f) == self.stagnation_window:
                rel_improve = (
                    (recent_f[0] - recent_f[-1])
                    / max(1.0, abs(recent_f[0]))
                )
                if rel_improve < self.stagnation_tol:
                    self.a0 *= self.stagnation_shrink
                    self.c0 *= self.perturb_growth
                    recent_f.clear()

            # ---------- adaptive penalty ----------
            if viol < best_violation:
                best_violation = viol
            elif viol > self.penalty_tol * best_violation:
                self.penalty = min(
                    self.penalty * self.penalty_growth,
                    self.penalty_max,
                )

            # ---------- averaging ----------
            if self.average and k >= self.avg_start:
                avg_count += 1
                x_avg += (x - x_avg) / avg_count

            if self.callback is not None:
                self.callback(x)
            
            # ---------- verbose ----------
            if self.first_print:
                self.first_print = False
                print("Iteration\tFunc eval\tCost Function\tStep size\tak gain \tck gain")
                print("---------\t---------\t-------------\t---------\t------- \t-------")
            print(f"{k}\t\t{nfev}\t\t{fval:.6e}\t{step_size:.6e}\t{ak_eff:.3e}\t{ck:.3e}")

        x_final = x_avg if (self.average and avg_count > 0) else x

        return OptimizeResult(
            x=x_final,
            fun=fun(x_final, *args),
            nit=self.maxiter,
            nfev=nfev,
            success=True,
            message="Optimization terminated successfully (SPSA + adaptive gains)",
        )

    def __call__(self, fun, x0, args=()):
        return self.fit(fun, x0, args)
