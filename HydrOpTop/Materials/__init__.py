#from .plot_function import plot_function
import importlib

__all__ = [
    "Log_SIMP",
    "RAMP",
    "Identity",
    "MultiMaterials",
]

def __getattr__(name):
    try:
        mod = importlib.import_module(f".{name.lower()}", __package__)
        obj = getattr(mod, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
