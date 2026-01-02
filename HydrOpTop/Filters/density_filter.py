from .Mesh_NNR import find_neighbors_within_radius2 as find_neighbors_within_radius
import numpy as np
from scipy.sparse import dia_matrix
from scipy.io import mmwrite
from .Base_Filter_class import Base_Filter

class Density_Filter(Base_Filter):
  r"""
  Description:
    Smooth the density parameter `p` at a given cell `i` according to its value
    at the neighboring cells `j` weighted by the distance of their respective 
    centers:

    .. math::
      
      \bar{p}_i = \frac{R^n V_i p_i + \sum_{j \in \partial i} (R-d_{ij})^n V_j p_j}
                        {R^n V_i + \sum_{j \in \partial i} (R-d_{ij})^n V_j}
      
    This filter was proposed by Bruns and Tortorelli (2001) and Bourdin (2001). 
    Usefull to remove numerical instability and deals with the checkboard effect.

  Parameters:
    ``filter_radius`` (float or list): the ball radius (float value) on which
    to search for neighboring cell center for averaging the density parameter.
    If a list is provided (i.e. ``[dx,dy,dz]``), the cell centers are searched
    into a ellipsoid of half axis dx, dy and dz.
    
    ``distance_weighting_power`` the exponent for distance weighting :math:`n` 
  
  Required solver outputs:
  
  """
  def __init__(self, cell_ids, radius=1., distance_weighting_power=1.):
    super(Density_Filter, self).__init__()
    if isinstance(cell_ids,str) and cell_ids == "__all__":
      self.cell_ids = None
    else:
      self.cell_ids = cell_ids
    self.input_ids = self.cell_ids
    self.output_ids = self.cell_ids
    self.filter_radius = radius
    if distance_weighting_power <= 0.: 
      print("distance_weighting_power argument must be strictly positive")
      raise ValueError
    self.distance_weighting_power = distance_weighting_power
    self.p_ids = None
    self.inputs = {}
    self.neighbors = None
    self.initialized = False
    
    self.variables_needed = ["ELEMENT_CENTER_X", "ELEMENT_CENTER_Y",
                            "ELEMENT_CENTER_Z", "VOLUME"]
    return
  
  def set_inputs(self, inputs):
    self.inputs = inputs
    self.inputs["ELEMENT_CENTER"] = np.array(
      [self.inputs["ELEMENT_CENTER_" + x] for x in "XYZ"]
    ).transpose()
    return
  
  
  def initialize(self):
    V = self.inputs["VOLUME"] #just need those in the optimized domain
    X = self.inputs["ELEMENT_CENTER"]
    if isinstance(self.filter_radius, list): #anisotropic
      for i in range(3): X[:,i] /= self.filter_radius[i]
      R = 1.
    else:
      R = self.filter_radius
    self.D_matrix = -find_neighbors_within_radius(X, R)
    self.D_matrix.data += R
    self.D_matrix.data = self.D_matrix.data ** self.distance_weighting_power
    self.D_matrix = self.D_matrix.dot( dia_matrix((V[np.newaxis,:],0),
                                                   shape=self.D_matrix.shape) )
    self.D_matrix = self.D_matrix.multiply(1/self.D_matrix.sum(axis=1))
    self.initialized = True
    return
  
  def get_filtered_density(self, p, p_bar=None):
    if not self.initialized: self.initialize()
    if p_bar is None:
      p_bar = np.zeros(len(p), dtype='f8')
    p_bar[:] = self.D_matrix @ p
    #print(self.inputs["ELEMENT_CENTER"][self.p_ids,:][self.D_matrix.getrow(2).tocoo().col])
    #temp = self.D_matrix.dot( dia_matrix((p[np.newaxis,:],0),shape=self.D_matrix.shape) )
    #p_bar[:] = (temp.sum(axis=1) / self.D_matrix.sum(axis=1)).flatten()
    return p_bar
  
  def get_filter_derivative(self, p):
    if not self.initialized: self.initialize()
    out = self.D_matrix
    return out


  def write_vtu_fitered_density(self, filename, p):
      """
      Write a VTU point dataset with volume and density parameters.

      Parameters
      ----------
      filename : str
          Output .vtu file path
      p : array_like, shape (N,)
          Density parameter associated with each point
      p_bar : array_like, shape (N,)
          Filtered density parameter (bijection output)
      """

      # Geometry: element centers
      points = self.inputs["ELEMENT_CENTER"]
      # Scalars
      volume = self.inputs["VOLUME"]
      p = np.asarray(p, dtype=float)
      p_bar = self.get_filtered_density(p)

      # Basic consistency checks
      n = points.shape[0]
      assert volume.shape[0] == n
      assert p.shape[0] == n
      assert p_bar.shape[0] == n

      # Create PolyData with vertex cells
      import pyvista as pv
      cloud = pv.PolyData(points)

      # Attach point data
      cloud.point_data["volume"] = volume
      cloud.point_data["p"] = p
      cloud.point_data["p_bar"] = p_bar

      # Write file
      if filename[:-4] != '.vtp': filename += '.vtp'
      cloud.save(filename)
      return


  @classmethod
  def sample_instance(cls):
    insts = []
    N = 100
    cell_ids = np.arange(N)
    rng = np.random.default_rng()
    # create test
    for p in [0.5, 1., 2.]:
      instance = cls(cell_ids, radius=0.1, distance_weighting_power=p)
      instance.inputs = {
        "ELEMENT_CENTER":rng.random((N,2)),
        "VOLUME":rng.random(N)
      }
      instance.input_indexes = cell_ids
      instance.output_indexes = cell_ids
      insts.append(instance)
    return insts

