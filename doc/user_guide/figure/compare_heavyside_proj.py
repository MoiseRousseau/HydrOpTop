import sys
sys.path.append('../../../')
sys.path.append('../../../verification/test_filter/')

from HydrOpTop.Filters import Heavyside_Filter
from HydrOpTop.Filters import Volume_Preserving_Heavyside_Filter
import matplotlib.pyplot as plt
import numpy as np

fig,ax = plt.subplots()

x = np.linspace(0,1,100)
h = Heavyside_Filter(0.2, 5)
hv = Volume_Preserving_Heavyside_Filter(0.2, 5)
yh = h.get_filtered_density(x)
yhv = hv.get_filtered_density(x)

ax.plot(x,yh,'b',label='Heavyside ref.')
ax.plot(x,yhv,'r',label='Volume preserving')
ax.grid()
ax.legend()

plt.show()
