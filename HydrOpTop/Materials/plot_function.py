import numpy as np


def plot_function(parametrization_class, block=True):
  """
  Description:
    Help visualize the transformation applied to the (filtered) density parameter the get the material properties.
    Require the ``matplotlib`` library to be installed.
  
  Parameter:
    ``parametrization_class`` (parametrization instance): the material parametrization instance as define above.
    
    ``block`` (bool): set to ``False``, does not block the GUI.
  
  """
  try:
    import matplotlib.pyplot as plt
  except:
    print("Plot requires the matplotlib library")
    return
  p = np.arange(0,101)/100
  K = parametrization_class.convert_p_to_mat_properties(p)
  dK = parametrization_class.d_mat_properties(p)
  
  fig, ax = plt.subplots()
  ax.plot(p,K,'r', label="value")
  ax2 = ax.twinx()
  ax2.plot(p,dK,'b',label="derivative")
  ax.set_xlabel("(Filtered) Density parameter p")
  ax.set_xlim([0,1])
  ax.set_ylabel(f"{parametrization_class.name}")
  ax2.set_ylabel(f"d {parametrization_class.name} / dp")
  ax.grid()
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1+h2, l1+l2)
  plt.tight_layout()
  plt.show(block=block)
  return ax
    

if __name__ == "__main__":
  import SIMP, Log_SIMP, RAMP
  mats = [SIMP.SIMP, Log_SIMP.Log_SIMP, RAMP.RAMP]
  for mat in mats:
    param = mat("all",mat.__name__,[0.1,1])
    ax = plot_function(param, block=False)
  input("Press a key to quit")

