import importlib
import inspect
import numpy as np
import pytest
import sys
import pkgutil

# --- Configuration ---
MODULE = "HydrOpTop.Functions"   # change to your actual module path
EPS = 1e-6
TOL = 1e-4


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


def get_classes_from_module(module_name, base_class_name="Base_Function"):
    """Import all classes from a package, skipping abstract base classes."""
    mod = importlib.import_module(module_name)

    # Load all submodules (handle lazy imports)
    if hasattr(mod, "__path__"):
        for _, sub_name, is_pkg in pkgutil.walk_packages(mod.__path__, mod.__name__ + "."):
            importlib.import_module(sub_name)

    # Find all classes defined in the module tree
    all_classes = []
    for m_name, m in list(sys.modules.items()):
        if m_name.startswith(module_name):
            for _, cls in inspect.getmembers(m, inspect.isclass):
                if cls.__module__.startswith(module_name):
                    all_classes.append(cls)

    # Filter: remove the base class and abstract ones
    concrete_classes = []
    for cls in all_classes:
        if cls.__name__.lower().startswith(base_class_name.lower()):
            # skip the base itself
            continue
        if not hasattr(cls, "evaluate") or not hasattr(cls, "d_objective"):
            # skip anything missing methods
            continue
        if inspect.isabstract(cls):
            continue
        concrete_classes.append(cls)

    return concrete_classes


@pytest.mark.parametrize("cls", list(get_classes_from_module(MODULE)))
def test_derivative_consistency(cls):
    """Check that analytical derivative matches finite difference."""

    # --- try to instantiate ---
    try:
        instance = cls()
    except Exception:
        # if class provides a 'sample_instance' or 'make_test_instance' helper, use it
        factory = None
        for name in ("sample_instance", "make_test_instance", "example"):
            if hasattr(cls, name) and callable(getattr(cls, name)):
                factory = getattr(cls, name)
                break
        if factory:
            instance = factory()
        else:
            pytest.skip(f"{cls.__name__}: cannot instantiate automatically")

    # --- check methods ---
    if not hasattr(instance, "evaluate") or not hasattr(instance, "d_objective"):
        pytest.skip(f"{cls.__name__}: missing evaluate or d_objective")

    # --- choose input ---
    if hasattr(instance, "input_size"):
        p = np.random.randn(instance.input_indexes)
    elif hasattr(instance, "sample_input") and callable(instance.sample_input):
        p = np.asarray(instance.sample_input())
    else:
        p = np.random.randn(3)

    # --- evaluate dp ---
    g_true = np.asarray(instance.d_objective_dp_partial(p))
    g_fd = finite_difference_dp(instance, p)
    diff = np.linalg.norm(g_true - g_fd)
    rel = diff / (np.linalg.norm(g_fd) + 1e-12)
    assert rel < TOL, (
        f"{cls.__name__} derivative dp mismatch:\n"
        f"analytic={g_true}\nfinite-diff={g_fd}\nrelerr={rel}"
    )

    # --- evaluate dvar ---
    vars = instance.variables_needed
    for var in vars:
        x = instance.inputs[var]
        g_true = np.asarray(instance.d_objective(var, x))
        g_fd = finite_difference_dvar(instance, var, p)

        # --- compare ---
        diff = np.linalg.norm(g_true - g_fd)
        rel = diff / (np.linalg.norm(g_fd) + 1e-12)
        assert rel < TOL, (
            f"{cls.__name__} derivative {var} mismatch:\n"
            f"analytic={g_true}\nfinite-diff={g_fd}\nrelerr={rel}"
        )
