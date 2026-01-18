import importlib
from .Base_Function_class import Base_Function
__all__ = [
    "Cell_Gradient",
    "Drawdown",
    "Least_Square_Calibration",
    "Mean_Liquid_Piezometric_Head",
    "Mechanical_Compliance",
    "p_Gradient",
    "p_Weighted_Cell_Gradient",
    "p_Weighted_Sum",
    "p_Weighted_Sum_Flux",
    "Reference_Liquid_Head",
    "Sum_Flux",
    "Sum_Variable",
    "Volume_Percentage",
]

def __getattr__(name):
    try:
        mod = importlib.import_module(f".{name.lower()}", __package__)
        obj = getattr(mod, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
