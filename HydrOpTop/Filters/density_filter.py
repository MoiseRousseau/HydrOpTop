from .Mesh_NNR import Mesh_NNR
import numpy as np
from scipy.sparse import dia_matrix
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
  def __init__(self, filter_radius=1., distance_weighting_power=1):
    self.filter_radius = filter_radius
    if distance_weighting_power <= 0.: 
      print("distance_weighting_power argument must be strictly positive")
      raise ValueError
    self.distance_weighting_power = distance_weighting_power
    self.p_ids = None
    self.inputs = {}
    self.neighbors = None
    self.initialized = False
    
    self.input_variables_needed = ["ELEMENT_CENTER_X", "ELEMENT_CENTER_Y",
                                   "ELEMENT_CENTER_Z", "VOLUME"]
    return
  
  def set_inputs(self, inputs):
    self.inputs = inputs
    self.inputs["ELEMENT_CENTER"] = np.array(
      [v for k,v in self.inputs.items() if "ELEMENT_CENTER_" in k]
    ).transpose()
    return
  
  
  def initialize(self):
    if self.p_ids is not None:
      V = self.inputs["VOLUME"][self.p_ids] #just need those in the optimized domain
      X = self.inputs["ELEMENT_CENTER"][self.p_ids,:]
    else:
      V = self.inputs["VOLUME"]
      X = self.inputs["ELEMENT_CENTER"]
    if isinstance(self.filter_radius, list): #anisotropic
      for i in range(3): X[:,i] /= self.filter_radius[i]
      R = 1.
    else:
      R = self.filter_radius
    print("Build kDTree and compute mesh fixed radius neighbors")
    self.neighbors = Mesh_NNR(X)
    self.neighbors.find_neighbors_within_radius(R)
    self.D_matrix = -self.neighbors.get_distance_matrix().tocsr(copy=True)
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

