#from .PFLOTRAN import PFLOTRAN
#from .FEFLOW import FEFLOW
#from .Linear_Elasticity_2D import Linear_Elasticity_2D
#from .Dummy_Simulator import Dummy_Simulator
#from .Base_Simulator import Base_Simulator

# Lazy importation
import importlib

__all__ = [
    "Dummy_Simulator",
    "FEFLOW",
    "HGS",
    "PFLOTRAN",
]  # Solvers you want to expose

def __getattr__(name):
    try:
        mod = importlib.import_module(f".{name.lower()}", __package__)
        obj = getattr(mod, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
