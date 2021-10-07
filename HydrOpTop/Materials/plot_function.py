import numpy as np


def plot_function(self, interpolation_class):
  try:
    import matplotlib.pyplot as plt
  except:
    print("Plot requires the matplotlib library")
    return
  p = np.arange(0,101)/100
  K = interpolation_class.convert_p_to_mat_properties(p)
  dK = interpolation_class.d_mat_properties(p)
  fig, ax = plt.subplots()
  ax.plot(p,K,'r', label="value")
  ax2 = ax.twinx()
  ax2.plot(p,dK,'b',label="derivative")
  ax.set_xlabel("Material parameter p")
  ax.set_xlim([0,1])
  ax.set_ylabel(f"{interpolation_class.name}")
  ax2.set_ylabel(f"d {interpolation_class.name} / dp")
  ax.grid()
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1+h2, l1+l2)
  plt.show()
  return
    


