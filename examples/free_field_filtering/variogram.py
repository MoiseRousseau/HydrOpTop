from HydrOpTop.Filters import Density_Filter
import utils
import numpy as np
import matplotlib.pyplot as plt

seeds = np.genfromtxt("seeds.txt")
bounding_box = [-0.0001,1.0001,-0.0001,1.0001]
mesh = utils.voronoi_bounded(seeds,bounding_box)
n_pts = len(seeds)

rng = np.random.default_rng(12345)
p = rng.random(len(seeds),dtype='f8')
filter = Density_Filter(np.arange(len(seeds)), radius=0.1)
filter.set_inputs({
    "ELEMENT_CENTER_X": seeds[:,0],
    "ELEMENT_CENTER_Y": seeds[:,1],
    "ELEMENT_CENTER_Z": np.zeros(n_pts,dtype='f8'),
    "VOLUME": mesh.areas
})

fig, axarr = plt.subplots(1,3)
mesh.plot(p, ax=axarr[0], show=False)
p_bar = filter.get_filtered_density(p)
mesh.plot(p_bar, ax=axarr[1], show=False)
filter.plot_isotropic_variogram(
    p, ax=axarr[2], mplargs={"label":"Raw density parameter"}
)
filter.plot_isotropic_variogram(
    p_bar, ax=axarr[2], mplargs={"label":"Filtered density parameter"}
)
plt.show()