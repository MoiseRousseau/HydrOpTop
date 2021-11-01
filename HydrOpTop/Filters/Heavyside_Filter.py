import numpy as np
from scipy.sparse import dia_matrix
from .Base_Filter_class import Base_Filter

class Heavyside_Filter(Base_Filter):
  """
  Filter the density paramater using a three field method according to
  a Heavyside function
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

