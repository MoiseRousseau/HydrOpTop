import numpy as np
import scipy.interpolate as sinter
from .Base_Filter_class import Base_Filter


class Pilot_Points(Base_Filter):
  r"""
  Dimensionnality reduction technique.
  
  :param control_point: 2D or 3D coordinates or the control (pilot) points
  :type control_point: numpy.ndarray
  :param parametrized_cells: Restrict the parametrization to the given cells
  :type control_point: iterable
  :param interpolator: Interpolator function to use. Built-in are SciPy
  `CloughTocher2DInterpolator`, `NearestNDInterpolator`, `LinearNDInterpolator`
  or `RBFInterpolator` (default). Can be a user provided callable which given
  the control point coefficient return another callable which evaluate the
  filtered density parameter at any coordinates given.
  :type interpolator: str or callable
  :param three_dim: Perform a 3D interpolation (default is 2D)
  :type three_dim: bool
  """
  def __init__(self, 
    control_points,
    parametrized_cells,
    interpolator="RBFInterpolator", 
    three_dim=False
  ):
    self.control_points = control_points
    self.parametrized_cells = parametrized_cells
    self.p_ids = None
    self.input_variables_needed = ["ELEMENT_CENTER_X","ELEMENT_CENTER_Y"]
    if three_dim:
      self.input_variables_needed += ["ELEMENT_CENTER_Z"]
    
    p_0 = np.zeros(len(control_points))
    if interpolator == "CloughTocher2DInterpolator":
      if three_dim:
        raise ValueError("CloughTocher2DInterpolator is not 3D interpolator")
      pass
    elif interpolator == "NearestNDInterpolator":
      self.interpolator = lambda p_0: sinter.NearestNDInterpolator(control_points,p_0)
    elif interpolator == "LinearNDInterpolator":
      self.interpolator = lambda p_0: sinter.LinearNDInterpolator(control_points,p_0,fill_value=0.5)
    elif interpolator == "RBFInterpolator":
      if three_dim:
        raise ValueError("RBFInterpolator is not 3D interpolator")
      self.interpolator = lambda p_0: sinter.RBFInterpolator(control_points, p_0)
    elif callable(interpolator):
      self.interpolator =  interpolator
    else:
      raise ValueError("Interpolator not recognized")
    self.initialized = False
    return
  
  
  def set_inputs(self, inputs):
    self.coords = np.array(inputs).T
    return

  
  def get_filtered_density(self, p, p_bar=None):
    """
    Project the reduced dimensionnaly index p to the simulation index p_bar
    """
    if p_bar is None:
      if self.parametrized_cells == "all":
          p_bar = np.zeros(self.coords.shape[0], dtype='f8')
      else:
          p_bar = np.zeros(len(self.parametrized_cells), dtype='f8')
    new_interpolator = self.interpolator(p)
    if self.parametrized_cells == "all":
      p_bar[:] = new_interpolator(self.coords)
    else:
      p_bar[:] = new_interpolator(self.coords[self.parametrized_cells])
    return p_bar

