import importlib
import inspect
import numpy as np
import pytest
import sys
import pkgutil
from scipy.sparse.linalg import norm as spnorm

# --- Configuration ---
MODULE = "HydrOpTop.Filters"   # change to your actual module path
EPS = 1e-6
TOL = 1e-6


def get_classes_from_module(module_name, base_class_name="Base_Filter"):
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
        if not hasattr(cls, "get_filtered_density") or not hasattr(cls, "get_filter_derivative"):
            # skip anything missing methods
            continue
        if inspect.isabstract(cls):
            continue
        concrete_classes.append(cls)

    return concrete_classes


def collect_test_cases():
    """Yield (cls, instance) tuples for all sample inputs of each class."""
    for cls in get_classes_from_module(MODULE):
        if cls.skip_test:
            continue
        # --- instantiate the class or multiple instances ---
        instances = []
        for name in ("sample_instance", "make_test_instance", "example"):
            if hasattr(cls, name) and callable(getattr(cls, name)):
                out = getattr(cls, name)()
                if isinstance(out, (list, tuple)):
                    instances.extend(out)
                else:
                    instances.append(out)
                break
        
        if not instances:
            try:
                instances.append(cls())
            except Exception as e:
                print(f"⚠️ Skipping {cls.__name__}: cannot instantiate ({e})")
                continue

        # --- now for each instance, collect input samples ---
        for i, inst in enumerate(instances):
            yield pytest.param(cls, inst, id=f"{cls.__name__}-inst{i}")


@pytest.mark.parametrize("cls,instance", collect_test_cases())
def test_derivative_consistency(cls,instance):
    """Check that analytical derivative matches finite difference."""

    # --- choose input ---
    p = np.random.random(len(instance.input_indexes))
    
    # var to skip:
    deriv_var_to_skip = []
    if hasattr(instance, "deriv_var_to_skip"):
        deriv_var_to_skip = instance.deriv_var_to_skip

    # --- evaluate dp ---
    g_fd = super(cls, instance).get_filter_derivative(p, eps=EPS, drop_tol=1e-6)
    g_true = instance.get_filter_derivative(p)
    diff = spnorm(g_true - g_fd)
    rel = diff / (spnorm(g_fd) + 1e-12)
    assert rel < TOL, (
        f"{cls.__name__} derivative dp mismatch:\n"
        f"analytic={g_true}\nfinite-diff={g_fd}\nrelerr={rel}"
    )

    # --- evaluate dvar ---
    # Filter suppose not being influence by solver variable
    # vars = instance.variables_needed
    # for var in vars:
    #     if var in deriv_var_to_skip: continue
    #     g_true = np.asarray(instance.d_objective(var, p))
    #     g_fd = finite_difference_dvar(instance, var, p)

    #     # --- compare ---
    #     diff = spnorm(g_true - g_fd)
    #     rel = diff / (spnorm(g_fd) + 1e-12)
    #     assert rel < TOL, (
    #         f"{cls.__name__} derivative {var} mismatch:\n"
    #         f"analytic={g_true}\nfinite-diff={g_fd}\nrelerr={rel}"
    #     )
