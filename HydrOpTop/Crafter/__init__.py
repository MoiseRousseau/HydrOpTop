import importlib

__all__ = [
    "Steady_State_Crafter",
]

from .steady_state_crafter import Steady_State_Crafter

def __getattr__(name):
    try:
        mod = importlib.import_module(f".{name.lower()}", __package__)
        obj = getattr(mod, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
