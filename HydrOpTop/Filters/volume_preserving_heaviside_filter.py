import numpy as np
from scipy.sparse import dia_matrix
from .Base_Filter_class import Base_Filter
from ..Functions.Volume_Percentage import Volume_Percentage
from scipy.optimize import root_scalar

class Volume_Preserving_Heaviside_Filter(Base_Filter):
  r"""
  Description:
    A variation of the previous filter.
    Apply the smooth Heaviside function to the density parameter with a given
    steepness and cutoff according to Xu et al. (2009):

    .. math::
      
      \tilde{p}_i = \left\{ 
         \begin{array}{ll}
         \eta \left[ e^{-\beta(1-\bar{p}_i/\eta)} - 
         (1-\frac{\bar{p}_i}{\eta}) e^{-\beta}\right] \quad \mbox{if} \quad \bar{p}_i<\eta \\
         (1-\eta) \left[ 1-e^{-\beta(\bar{p}_i-\eta)/(1-\eta)} + 
         \frac{\bar{p}_i-\eta}{1-\eta}e^{-\beta} \right] + \eta \quad \mbox{else}
         \end{array} \\ \right.

  Parameters:
    ``cutoff`` (float): the cutoff parameter :math:`\eta` (i.e. the value of :math:`p_i` where the step is located)
    
    ``steepness`` (float): the steepness of the smooth Heaviside function :math:`\beta`. 
  
  
  Required solver outputs:
    ``None``
  
  This filter was proposed by Xu et al. (2009). See the original publication
  for more detail (https://link.springer.com/article/10.1007/s00158-009-0452-7)
  
  """
  def __init__(self, cutoff=0.5, steepness = 5, vol_constraint=None):
    self.cutoff = cutoff
    self.stepness = steepness
    self.vol_constraint = vol_constraint
    self.last_p = None
    return
  
  def get_filtered_density(self, p, p_filtered=None):
    if self.last_p is None:
      self.last_p = p.copy()
    else:
      self.last_p[:] = p
    if p_filtered is None:
      p_filtered = np.zeros(len(p), dtype='f8')
    else:
      p_filtered[:] = 0.
    cpb = 1 - p / self.cutoff
    pbr = (p-self.cutoff) / (1-self.cutoff)
    p_filtered[:] = np.where(p<=self.cutoff,
        self.cutoff*(np.exp(-self.stepness*cpb) - cpb*np.exp(-self.stepness)),
        (1-self.cutoff)*(1-np.exp(-self.stepness*pbr)+np.exp(-self.stepness)*pbr)+self.cutoff)
    
    return p_filtered
  
  def get_filter_derivative(self, p):
    self.last_p[:] = p
    d_p_filtered = np.where(p<=self.cutoff,
                     np.exp(-self.stepness * (1-p/self.cutoff) ),
                     np.exp(-self.stepness * (p-self.cutoff) / (1-self.cutoff) ) )
    d_p_filtered *= self.stepness
    d_p_filtered += np.exp(-self.stepness)
    d_p = dia_matrix((d_p_filtered[np.newaxis,:],0),
                                                   shape=(len(p),len(p)) )
    return d_p
  
  def update_stepness(self, stepness, ref_vol=None, autocorrect=True):
    print('\nUpdate Volume Preserving Heaviside Filter')
    print(f"Update from {self.stepness} to {stepness}")
    if not isinstance(self.vol_constraint, Volume_Percentage):
      print("Error, constraint must be of type Volume_Percentage")
      raise TypeError
    if ref_vol is None: 
      ref_vol = self.vol_constraint.evaluate(self.get_filtered_density(self.last_p))
      if ref_vol > 0.:
        print("Warning! Constraint not respected, solver may fail")
        print("Autocorrect this. To disable this feature, set the boolean autocorrect to False")
        ref_vol=-0.
      
    old_cutoff = self.cutoff
    self.stepness = stepness
    
    def func(n):
      self.cutoff = n
      p_filtered = self.get_filtered_density(self.last_p)
      diff = ref_vol - self.vol_constraint.evaluate(p_filtered)
      return diff 
    
    root_scalar(func, bracket=[0.01,0.99], x0 = self.cutoff)
    print(f"Old cutoff: {old_cutoff}")
    print(f"New cutoff: {self.cutoff}")
    return
    
  
  def plot_filtered_density(self):
    try:
      import matplotlib.pyplot as plt
    except:
      print("Matplotlib is not available on your installation")
      print("Please try 'pip3 install matplotlib' and restart the optimization")
    x = np.linspace(0,1,1000)
    y = self.get_filtered_density(x)
    fig,ax = plt.subplots()
    ax.plot(x,y,'b',label="Filtered parameter")
    ax.set_xlabel("Input Parameter")
    ax.set_ylabel("Filtered Parameter")
    ax.grid()
    plt.show()
    return
  

