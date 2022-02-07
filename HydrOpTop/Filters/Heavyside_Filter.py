import numpy as np
from scipy.sparse import dia_matrix
from .Base_Filter_class import Base_Filter

class Heavyside_Filter(Base_Filter):
  r"""
  Decription:
    Apply the smooth Heavyside function to the density parameter with a given
    steepness and cutoff. Heavyside filters are usually applied on already filtered
    field using the previous density filter to help impose a minimum length constrain
    and to avoid blurry contour. The smooth Heavyside function is defined as:

    .. math::
      
       \tilde{p}_i = \frac{\tanh(\beta \eta) + \tanh(\beta (\bar p - \eta))}
                          {\tanh(\beta \eta) + \tanh(\beta (1- \eta))}

  Parameters: 
    ``cutoff`` (float): the cutoff parameter :math:`\eta` (i.e. the value of 
    :math:`p_i` where the step is located)
    
    ``steepness`` (float): the steepness of the smooth Heavyside function :math:`\beta`
  
  Required solver output:

  """
  def __init__(self, cutoff=0.5, steepness = 5):
    self.cutoff = cutoff
    self.stepness = steepness
    self.initialized = False
    self.name = "Heavyside Filter"
    return
  
  def get_filtered_density(self, p):
    a = np.tanh(self.stepness*self.cutoff)
    p_filtered = (a+np.tanh(self.stepness*(p-self.cutoff))) / \
                       (a+np.tanh(self.stepness*(1-self.cutoff)))
    return p_filtered
  
  def get_filter_derivative(self, p):
    d_p_filtered = 1 - np.tanh(self.stepness*(p-self.cutoff))**2
    a = np.tanh(self.stepness*self.cutoff)
    d_p_filtered *= self.stepness / (a + np.tanh(self.stepness*(1-self.cutoff)))
    return dia_matrix((d_p_filtered,[0]), shape=(len(p),len(p)))
  
  def plot_filtered_density(self):
    r"""
    Visualize the transformation applied
    
    """
    try:
      import matplotlib.pyplot as plt
    except:
      print("Matplotlib is not available on your installation")
      print("Please try 'pip3 install matplotlib' and restart the optimization")
    x = np.linspace(0,1,1000)
    save = self.base_density_filter
    self.base_density_filter = None
    y = self.get_filtered_density(x)
    fig,ax = plt.subplots()
    ax.plot(x,y,'b',label="Filtered parameter")
    ax.set_xlabel("Input Parameter")
    ax.set_ylabel("Filtered Parameter")
    ax.grid()
    plt.show()
    self.base_density_filter = save
    return

