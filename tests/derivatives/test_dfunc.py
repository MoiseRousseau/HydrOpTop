import importlib
import inspect
import numpy as np
import pytest
import sys
import pkgutil
from utils import finite_difference_dp, finite_difference_dvar

# --- Configuration ---
MODULE = "HydrOpTop.Functions"   # change to your actual module path
EPS = 1e-6
TOL = 1e-4


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


def collect_test_cases():
    """Yield (cls, instance) tuples for all sample inputs of each class."""
    for cls in get_classes_from_module(MODULE):
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
            try: instances.append(cls())
            except:
                print("Can add", cls)

        # --- now for each instance, collect input samples ---
        for i, inst in enumerate(instances):
            yield pytest.param(cls, inst, id=f"{cls.__name__}-inst{i}")


@pytest.mark.parametrize("cls,instance", list(collect_test_cases()))
def test_derivative_consistency(cls,instance):
    """Check that analytical derivative matches finite difference."""
    print(cls, instance.inputs)

    # --- check methods ---
    if not hasattr(instance, "evaluate") or not hasattr(instance, "d_objective"):
        pytest.skip(f"{cls.__name__}: missing evaluate or d_objective")

    # --- choose input ---
    if hasattr(instance, "input_indexes"):
        if instance.input_indexes is None:
            p = [1] # scalar so act as a array with infinite length
        else:
            p = np.random.rand(len(instance.input_indexes))
    elif hasattr(instance, "sample_input") and callable(instance.sample_input):
        p = np.asarray(instance.sample_input())
    else:
        p = np.random.rand(3)
    
    # var to skip:
    deriv_var_to_skip = []
    if hasattr(instance, "deriv_var_to_skip"):
        deriv_var_to_skip = instance.deriv_var_to_skip

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
        if var in deriv_var_to_skip: continue
        g_true = np.asarray(instance.d_objective(var, p))
        g_fd = finite_difference_dvar(instance, var, p)

        # --- compare ---
        diff = np.linalg.norm(g_true - g_fd)
        rel = diff / (np.linalg.norm(g_fd) + 1e-12)
        assert rel < TOL, (
            f"{cls.__name__} derivative {var} mismatch:\n"
            f"analytic={g_true}\nfinite-diff={g_fd}\nrelerr={rel}"
        )
