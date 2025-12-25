# more density filter in https://doi.org/10.1007%2Fs00158-006-0087-x
# more in https://www.sciencedirect.com/science/article/pii/S0965997816302174
# and https://link.springer.com/article/10.1007/s00158-009-0452-7
# more more https://link.springer.com/article/10.1007/s00158-019-02194-x


import importlib

__all__ = [
    "Filter_Sequence",
    "Density_Filter",
    "Heaviside_Filter",
    #"No_Filter",
    "Pilot_Points",
    "Volume_Preserving_Heaviside_Filter",
    "Zone_Homogeneous",
]

def __getattr__(name):
    try:
        mod = importlib.import_module(f".{name.lower()}", __package__)
        obj = getattr(mod, name)
        return obj
    except (ModuleNotFoundError, AttributeError) as e:
        raise AttributeError(f"module {__name__} has no attribute {name}") from e
