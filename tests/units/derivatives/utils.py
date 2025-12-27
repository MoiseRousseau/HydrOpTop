import numpy as np
EPS=1e-6

def finite_difference_dvar(f, var, p, eps=EPS):
    """Compute numerical gradient via central difference."""
    x = f.inputs[var]
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        f.inputs[var] = x1
        f1 = f.evaluate(p)
        f.inputs[var] = x2
        f2 = f.evaluate(p)
        grad[i] = (f1 - f2) / (2 * eps)
    return grad

def finite_difference_dp(f, p, eps=EPS):
    grad = np.zeros_like(p)
    for i in range(len(p)):
        x1 = p.copy()
        x2 = p.copy()
        x1[i] += eps
        x2[i] -= eps
        grad[i] = (f.evaluate(x1) - f.evaluate(x2)) / (2 * eps)
    return grad