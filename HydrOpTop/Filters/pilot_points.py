import numpy as np
import scipy.interpolate as sinter
from .Base_Filter_class import Base_Filter


class Pilot_Points(Base_Filter):
    """
    Dimensionality reduction technique using spatial interpolation.

    This function applies a dimensionality reduction method based on a set
    of control (pilot) points and an interpolation technique. It supports
    both 2D and 3D cases and allows for restricting the operation to specific
    cells or regions.

    Parameters
    ----------
    control_point : numpy.ndarray
        A NumPy array of shape (N, 2) or (N, 3) containing the coordinates
        of the control (pilot) points.

    parametrized_cells : iterable
        An iterable (e.g., list or array) of cell indices or identifiers
        to which the parametrization should be restricted.

    interpolator : str or callable
        The interpolation method to use. Can be one of the built-in strings:
        `'CloughTocher2DInterpolator'`, `'NearestNDInterpolator'`,
        `'LinearNDInterpolator'`, or `'RBFInterpolator'` (default).
        Alternatively, a user-provided callable can be passed, which should
        accept control point coefficients and return a function that evaluates
        the filtered density parameter at arbitrary coordinates.

    three_dim : bool, optional
        If True, perform 3D interpolation. If False (default), use 2D interpolation.

    Returns
    -------
    <Specify return type>
        <Brief description of the return value(s)>

    Raises
    ------
    <ExceptionType>
        <Condition under which the exception is raised>

    Examples
    --------
    >>> control_points = np.array([[0, 0], [1, 1], [2, 2]])
    >>> reduce(control_point=control_points, interpolator="RBFInterpolator")

    Notes
    -----
    Make sure the number and distribution of control points is adequate
    for the chosen interpolation method to avoid numerical instability.
    """
    def __init__(self,
      control_points,
      parametrized_cells,
      interpolator="RBFInterpolator",
      three_dim=False
    ):
      super(Pilot_Points, self).__init__()
      self.control_points = control_points
      self.parametrized_cells = parametrized_cells
      self.input_ids = self.parametrized_cells[:len(control_points)]
      self.output_ids = self.parametrized_cells
      self.variables_needed = ["ELEMENT_CENTER_X","ELEMENT_CENTER_Y"]
      self.dim = 3 if three_dim else 2
      if three_dim:
        self.variables_needed += ["ELEMENT_CENTER_Z"]

      p_0 = np.zeros(len(control_points))+0.5
      if interpolator == "CloughTocher2DInterpolator":
        if three_dim:
          raise ValueError("CloughTocher2DInterpolator is not 3D interpolator")
        pass
      elif interpolator == "NearestNDInterpolator":
        self.interpolator = sinter.NearestNDInterpolator(control_points,p_0)
      elif interpolator == "LinearNDInterpolator":
        self.interpolator = sinter.LinearNDInterpolator(control_points,p_0,fill_value=0.5)
      elif interpolator == "RBFInterpolator":
        if three_dim:
          raise ValueError("RBFInterpolator is not 3D interpolator")
        self.interpolator = sinter.RBFInterpolator(control_points, p_0)
      elif callable(interpolator):
        self.interpolator =  interpolator
      else:
        raise ValueError("Interpolator not recognized")
      self.initialized = False
      self.coords = None
      return


    def set_inputs(self, inputs):
      super(Pilot_Points, self).set_inputs(inputs)
      if self.coords is None:
        self.coords = np.array(
          [self.inputs[k] for k in self.variables_needed]
        ).transpose()
      return


    def get_filtered_density(self, p):
      """
      Project the reduced dimensionnaly index p to the simulation index p_bar
      """
      if isinstance(self.interpolator, sinter.RBFInterpolator):
        self.interpolator = sinter.RBFInterpolator(self.control_points, p)
      else:
        if len(self.interpolator.values.shape) != 1:
          self.interpolator.values[:,0] = p
        else:
          self.interpolator.values[:] = p
      p_bar = self.interpolator(self.coords[:,:self.dim])
      return p_bar

    def get_input_ids(self):
      return self.output_ids

    @classmethod
    def sample_instance(cls):
      N = 10
      mesh_cell_center = np.random.random((N,3)) #3D mesh
      ctrlp = np.array([[0,0],[1,0],[0,1],[1,1]],dtype='f8')
      p_cell = np.arange(0,N) # parametrize all cell
      possible_interpolators = [
        "NearestNDInterpolator",
        "LinearNDInterpolator",
        "RBFInterpolator"
      ]
      # create test
      insts = []
      for interpolator in possible_interpolators:
        instance = cls(ctrlp, p_cell, interpolator=interpolator)
        instance.coords = mesh_cell_center
        instance.input_indexes = np.arange(len(ctrlp))
        instance.output_indexes = np.arange(len(mesh_cell_center))
        insts.append(instance)
      return insts
