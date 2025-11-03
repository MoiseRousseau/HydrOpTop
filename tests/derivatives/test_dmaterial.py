import importlib
import inspect
import numpy as np
import pytest
import sys
import pkgutil

# --- Configuration ---
MODULE = "HydrOpTop.Materials"   # change to your actual module path
EPS = 1e-6
TOL = 1e-6



def get_classes_from_module(module_name, base_class_name="Base_Material"):
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
        if inspect.isabstract(cls):
            continue
        concrete_classes.append(cls)

    return concrete_classes


def collect_test_cases():
    """Yield (cls, instance) tuples for all sample inputs of each class."""
    N = 10
    default_args = {
        "cell_ids_to_parametrize":"__all__",
        "property_name":"TEST",
        "bounds":[0.1,1],
    }
    
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
                instances.append(cls(**default_args))
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
    p = np.random.random(5)

    # --- evaluate dp ---
    g_fd = super(cls, instance).d_mat_properties(p, eps=EPS) # base class did it by FD
    g_true = instance.d_mat_properties(p)
    diff = np.linalg.norm(g_true - g_fd)
    rel = diff / (np.linalg.norm(g_fd) + 1e-12)
    assert rel < TOL, (
        f"{cls.__name__} derivative dp mismatch:\n"
        f"analytic={g_true}\nfinite-diff={g_fd}\nrelerr={rel}"
    )