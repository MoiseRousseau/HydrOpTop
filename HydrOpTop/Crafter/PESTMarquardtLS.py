"""
PEST-style Levenberg–Marquardt with λ-line-search and bounds.
"""

import numpy as np


def _sigmoid(u):
    out = np.empty_like(u)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    expu = np.exp(u[~pos])
    out[~pos] = expu / (1 + expu)
    return out

def _sigmoid_der(u):
    s = _sigmoid(u)
    return s * (1 - s)


class PESTMarquardtLS:
    """
    PEST-style Marquardt (Levenberg-Marquardt) fitter with box-bounds via smooth transform.
    Allow a user-specified λ-scan, solver picks the λ that produces lowest cost.

    Usage:
      - residuals_fn(p) -> r   (array of residuals, length m)
      - optionally jacobian_fn(p) -> J  shape (m, n)
      - call `fit(residuals_fn, x0, bounds=(lower, upper), jac=jacobian_fn, ...)`
    """

    def __init__(self,
                 maxiter: int = 200,
                 ftol: float = 1e-8,
                 xtol: float = 1e-8,
                 gtol: float = 1e-8,
                 epsfcn: float = 1e-8,
                 lambda_init: float = 1e-2,
                 lambda_scale_up: float = 10.0,
                 lambda_scale_down: float = 0.1,
                 lambda_factors=(0.07, 0.12, 0.7, 4, 16, 64, 512),
                 verbose: bool = False):

        self.maxiter = maxiter
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.epsfcn = epsfcn
        self.lambda_init = lambda_init
        self.lambda_scale_up = float(lambda_scale_up)
        self.lambda_scale_down = float(lambda_scale_down)
        self.lambda_factors = list(lambda_factors)
        self.verbose = verbose

    # --- mapping u -> p with bounds -------------------
    @staticmethod
    def _map_u_to_p(u, lo, hi, mask):
        p = u.copy()
        dpdu = np.ones_like(u)
        if np.any(mask):
            uu = u[mask]
            s = _sigmoid(uu)
            p[mask] = lo[mask] + (hi[mask] - lo[mask]) * s
            dpdu[mask] = (hi[mask] - lo[mask]) * _sigmoid_der(uu)
        return p, dpdu

    # --- finite difference J wrt p --------------------
    def _fd_J(self, rfun, p, eps):
        r0 = rfun(p)
        m = len(r0)
        n = len(p)
        J = np.zeros((m, n))
        for j in range(n):
            h = eps * max(1.0, abs(p[j]))
            p2 = p.copy()
            p2[j] += h
            J[:, j] = (rfun(p2) - r0) / h
        return J

    # ---------------------------------------------------
    def fit(self, rfun, x0, bounds=None, jac=None):

        x0 = np.asarray(x0, float)
        n = len(x0)

        if bounds is None:
            lo = np.full(n, -np.inf)
            hi = np.full(n, +np.inf)
        else:
            lo = np.asarray(bounds[0], float)
            hi = np.asarray(bounds[1], float)

        # mask of bounded parameters
        mask = np.isfinite(lo) & np.isfinite(hi)

        # build u0 from x0
        u = x0.copy()
        if np.any(mask):
            s0 = (x0[mask] - lo[mask]) / (hi[mask] - lo[mask])
            eps = np.finfo(float).eps * 10
            s0 = np.clip(s0, eps, 1 - eps)
            u[mask] = np.log(s0 / (1 - s0))

        # evaluate initial
        p, dpdu = self._map_u_to_p(u, lo, hi, mask)
        r = rfun(p)
        cost = 0.5 * np.dot(r, r)

        lam = self.lambda_init
        cost_history = [cost]
        x_history = [p.copy()]

        if self.verbose:
            print(f"iter 0 cost={cost}")

        # --- iteration ---------------------------------
        for it in range(1, self.maxiter + 1):

            # Jacobian wrt p
            if jac is None:
                Jp = self._fd_J(rfun, p, self.epsfcn)
            else:
                Jp = jac(p)

            # chain rule to unconstrained u
            J = Jp * dpdu

            g = J.T @ r
            if np.linalg.norm(g, np.inf) < self.gtol:
                return self._finish(p, True, "gtol reached", it,
                                    cost_history, x_history)

            H = J.T @ J

            # -----------------------------------------------------
            # λ-line-search: build candidate λ_i and evaluate all
            # -----------------------------------------------------
            best_cost = np.inf
            best = None

            for fac in self.lambda_factors:
                lam_i = lam * fac

                diagH = np.diag(H).copy()
                diagH[diagH == 0.0] = 1.0

                A = H + lam_i * np.diag(diagH)
                b = -g
                du, *_ = np.linalg.lstsq(A, b, rcond=None)
                # try:
                #     du = np.linalg.solve(A, b)
                # except np.linalg.LinAlgError:
                #     du, *_ = np.linalg.lstsq(A, b, rcond=None)

                if np.linalg.norm(du) < self.xtol * (1 + np.linalg.norm(u)):
                    continue

                u_i = u + du
                p_i, dpdu_i = self._map_u_to_p(u_i, lo, hi, mask)
                r_i = rfun(p_i)
                cost_i = 0.5 * np.dot(r_i, r_i)

                if self.verbose:
                    print(f"Test upgrade λ={lam_i:.3g}, cost={cost_i}")

                if cost_i < best_cost:
                    best_cost = cost_i
                    best = (lam_i, u_i, p_i, dpdu_i, r_i, cost_i)

            if best is None:
                return self._finish(p, False, "All λ failed", it,
                                    cost_history, x_history)

            lam, u_new, p_new, dpdu_new, r_new, cost_new = best

            # accept the best
            u, p, dpdu, r, cost = u_new, p_new, dpdu_new, r_new, cost_new

            cost_history.append(cost)
            x_history.append(p.copy())

            if self.verbose:
                print(f"iter {it} finished: λ={lam:.3g} cost={cost}")

            # stopping on function change
            if abs(cost_history[-2] - cost_history[-1]) < self.ftol * (1 + cost):
                return self._finish(p, True, "ftol reached", it,
                                    cost_history, x_history)

        return self._finish(p, False, "maxiter reached", self.maxiter,
                            cost_history, x_history)

    # ---------------------------------------------------
    def _finish(self, p, success, msg, it, cost_hist, x_hist):
        return {
            "x": p,
            "success": success,
            "message": msg,
            "niter": it,
            "cost_history": cost_hist,
            "x_history": x_hist
        }
