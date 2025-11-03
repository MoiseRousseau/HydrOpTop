# This file is a part of the topology optimization program by Moise Rousseau

import numpy as np
from .Base_Function_class import Base_Function
from .function_utils import df_dX

class Reference_Liquid_Head(Base_Function):
  r"""
  Compute the difference between the head at the given cells and the head in the simulation.
  
  Note the cell id are 1 based. First cell given is ID 1.
  
  Required ``LIQUID_HEAD`` output.
  
  :param head: Reference head to compute the difference with
  :type head: iterable
  :param weights: Weights associated with each observation
  :type weights: iterable (same size as head)
  :param cell_ids: The corresponding cell ids of the head. If None, consider head[0] for cell id 1, head[1] for cell id 2, ...
  :type cell_ids: iterable
  :param observation_name: If observation points have name in the simulator, provide it here
  :type observation_name: list of str
  :param norm: Norm to compute the difference (i.e. 1 for sum of head error, 2 for MSE, inf for max difference
  :type norm: int
  """
  def __init__(
    self,
    head,
    cell_ids=None,
    XYZ_coordinates=None,
    time_obs=None,
    weights=None,
    observation_name=None,
    outlier_weaken=False,
    norm = 1,
  ):
    super(Reference_Liquid_Head, self).__init__()
    
    self.set_error_norm(norm)
    self.ref_head = np.array(head)
    self.weights = 1. if weights is None else np.array(weights)
    self.cell_ids = np.asarray(cell_ids) if cell_ids is not None else None
    self.XYZ = np.asarray(XYZ_coordinates) if XYZ_coordinates is not None else None
    if self.cell_ids is None and self.XYZ is None:
      raise ValueError("Both cell_ids and XYZ coordinates parameter cannot be null at the same time. If you passed it by name, please use solver function to get either cell_ids or coordinate and retry.")
    if (self.cell_ids is None) == (self.XYZ is None):
      print("Both cell_ids and coordinates passed, rely only on cell_ids...")
    self.time = time_obs
    self.observation_name = observation_name
    self.outlier = outlier_weaken
    self.h_interpolator = None

    # for plotting
    self.residuals = None

    #required for problem crafting
    self.variables_needed = ["LIQUID_HEAD"]
    if self.XYZ is not None and self.cell_ids is None:
      self.variables_needed = ["LIQUID_HEAD_INTERPOLATOR"]
    #if self.observation_name is not None:
    # 	self.solved_variables_needed = ["LIQUID_HEAD_AT_OBSERVATION"]
    self.name = "Reference Head"
    return
  
  def set_error_norm(self, x):
    if int(x) != x or x <= 0:
      raise ValueError("Error norm need to be a positive integer")
    self.norm = x
    return

  
  ### COST FUNCTION ###
  def __evaluate_cell_ids__(self,p):
    self.head = self.inputs["LIQUID_HEAD"]
    if self.observation_name is not None:
      self.residuals = np.array([
        self.head[x] - h for x,h in zip(self.observation_name, self.ref_head)
      ])
    if self.cell_ids is None: 
      self.residuals = self.head-self.ref_head
    else:
      self.residuals = self.head[self.cell_ids-1]-self.ref_head
    return np.sum(self.weights * self.residuals**self.norm)
    
  def __evaluate_xyz__(self,p):
    self.h_interpolator = self.inputs["LIQUID_HEAD_INTERPOLATOR"]
    heads = self.h_interpolator(self.XYZ)
    self.residuals = heads - self.ref_head
    return np.sum(self.weights * self.residuals**self.norm)

  def evaluate(self, p):
    """
    Evaluate the cost function
    Return a scalar of dimension [L]
    """
    if self.cell_ids is not None:
      res = self.__evaluate_cell_ids__(p)
    else:
      res = self.__evaluate_xyz__(p)
    return res
  
  
  ### PARTIAL DERIVATIVES ###
  def __d_objective_dh_cell_ids__(self, p):
    """
    Given a variable, return the derivative of the cost function according to that variable.
    Use current simulation state
    """
    head = self.inputs["LIQUID_HEAD"]
    # Derivative according to the head
    if self.observation_name is not None:
      r = np.array([
  head[x] - h for x,h in zip(self.observation_name, self.ref_head)
      ])
      dobj = self.norm * self.weights * r**(self.norm-1)
    elif self.cell_ids is None:
      dobj = self.norm * self.weights * (head-self.ref_head)**(self.norm-1)
    else:
      dobj = np.zeros(len(head), dtype='f8')
      dobj[self.cell_ids-1] = self.norm * self.weights * (
        head[self.cell_ids-1]-self.ref_head
      )**(self.norm-1)
    return dobj

  def __d_objective_dh_xyz__(self, p):
    """
    Given a variable, return the derivative of the cost function according to that variable.
    Use current simulation state
    """
    # update interpolator
    self.h_interpolator = self.inputs["LIQUID_HEAD_INTERPOLATOR"]
    heads = self.h_interpolator(self.XYZ)
    residuals = heads - self.ref_head

    pre = self.norm * self.weights * residuals**(self.norm-1)
    dobj = np.sum([
      pre[i] * df_dX(self.h_interpolator, xyz) for i,xyz in enumerate(self.XYZ)
    ],axis=0)
    return dobj


  def d_objective(self, var, p):
    """
    Given a variable, return the derivative of the cost function according to that variable.
    Use current simulation state
    """
    if self.cell_ids is not None and var == "LIQUID_HEAD":
      res = self.__d_objective_dh_cell_ids__(p)
    elif var == "LIQUID_HEAD_INTERPOLATOR":
      res = self.__d_objective_dh_xyz__(p)
    else:
      # The function depends of no other variables
      res = np.zeros_like(p, dtype='f8')
    return res


  def plot_scatter(self):
    """
    Draw a scatter plot of measured vs simulated head
    """
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    predicted_head = self.ref_head - self.residuals
    ax.scatter(
      self.ref_head, predicted_head,
      c='b'
    )
    hmin = np.min([self.ref_head, predicted_head])
    hmax = np.max([self.ref_head, predicted_head])
    a = 0.1
    range = hmax - hmin
    ax.plot([hmin - range*a, hmax + range*a],[hmin - range*a, hmax + range*a], c='k')
    ax.grid()
    ax.set_xlabel("Measured head")
    ax.set_ylabel("Predicted head")
    ax.set_xlim([hmin - range*a, hmax + range*a])
    ax.set_ylim([hmin - range*a, hmax + range*a])
    plt.tight_layout()
    plt.show()
    return


  @classmethod
  def sample_instance(cls):
    # sample cell_ids
    res1 = cls(head=[11.2,43.2,56.4,29.4], cell_ids=[2,4,5,10])
    res1.set_inputs({"LIQUID_HEAD":np.random.rand(20)[res1.cell_ids]*100})
    # sample xyz
    from scipy.interpolate import LinearNDInterpolator
    res2 = cls(
      head=[11.2,43.2,56.4,29.4],
      XYZ_coordinates=np.random.rand(4,3),
    )
    res2.set_inputs({
      "LIQUID_HEAD_AT_VERTICE":np.random.rand(20)*100,
      "MESH_VERTICE_XYZ": np.random.rand(20,3)*3-1
    })
    res2.deriv_var_to_skip = ["MESH_VERTICE_XYZ"]
    return [res1,res2]
