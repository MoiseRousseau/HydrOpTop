from .Mesh_NNR import Mesh_NNR
import numpy as np


class Density_Filter:
  """
  Filter the density parameter according to Bruns and Tortorelli (2001) and 
  Bourdin (2001):
  https://doi.org/10.1016%2FS0045-7825%2800%2900278-4
  https://doi.org/10.1002%2Fnme.116
  Summarized in:
  https://link.springer.com/article/10.1007/s00158-009-0452-7
  
  """
  def __init__(self, filter_radius=-1.):
    self.filter_radius = filter_radius
    self.p_ids = None
    self.volume = None
    self.mesh_center = None
    self.neighbors = None
    self.initialized = False
    
    self.output_variable_needed = ["X_COORDINATE", "Y_COORDINATE",
                                   "Z_COORDINATE", "VOLUME"]
    return
  
  def set_p_to_cell_ids(self, p_ids):
    self.p_ids = p_ids #if None, this mean all the cell are parametrized
    return
  
  def set_inputs(self, inputs):
    if self.p_ids is None:
      self.volume = inputs[3]
      self.mesh_center = np.array(inputs[:3]).transpose()
    else:
      self.volume = inputs[3][self.p_ids-1] #just need those in the optimized domain
      self.mesh_center = np.array(inputs[:3])[:,self.p_ids-1].transpose() #same here
    return
  
  def initialize(self):
    print("Build kDTree and compute mesh fixed radius neighbors")
    self.neighbors = Mesh_NNR(self.mesh_center)
    self.neighbors.find_neighbors_within_radius(self.filter_radius)
    self.initialized = True
    return
  
  def get_filtered_density(self, p, p_bar=None):
    if not self.initialized: self.initialize()
    if p_bar is None:
      p_bar = np.zeros(len(p), dtype='f8')
    for i in range(len(p_bar)):
      indices, distances = self.neighbors.get_neighbors_center(i)
      temp = (self.filter_radius - distances) * self.volume[indices]
      p_bar[i] = temp.dot(p[indices]) / np.sum(temp)
    return p_bar
  
  def get_filter_derivative(self, p, out=None):
    if not self.initialized: self.initialize()
    if out is None:
      out = self.neighbors.get_distance_matrix().copy()
      
    num_sum_neighbors = np.zeros(len(p), dtype='f8')
    for i in range(len(num_sum_neighbors)):
      indices, distances = self.neighbors.get_neighbors_center(i)
      num_sum_neighbors[i] = np.sum( (self.filter_radius - distances) * \
                                     self.volume[indices] )
      
    distances = self.neighbors.get_distance_matrix().todok()
    count = 0
    #populate matrix row per row
    for indices, distance in distances.items():
      out.data[count] = (self.filter_radius - distance) * self.volume[indices[1]] / \
                         num_sum_neighbors[indices[0]]
      count += 1
    return out
  
  def __get_PFLOTRAN_output_variable_needed__(self):
    return self.output_variable_needed 
  

