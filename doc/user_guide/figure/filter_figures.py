import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../')
sys.path.append('../../../tests/filters/')

def density_filter_fig():
  from HydrOpTop.Filters import Density_Filter
  import utils
  seeds = np.genfromtxt("seeds.txt")
  group1 = np.genfromtxt("group1.txt",dtype='i8')
  group2 = np.genfromtxt("group2.txt",dtype='i8')
  fig,axarr = plt.subplots(2,2, sharex=True, sharey=True)
  bounding_box = [-0.0001,1.0001,-0.0001,1.0001]
  mesh = utils.voronoi_bounded(seeds,bounding_box)
  n_pts = len(seeds)
  p_ref = np.zeros(len(seeds),dtype='f8')
  p_ref[group1-1] = 1.
  p_ref[group2-1] = 1.
  filter = Density_Filter()
  filter.set_p_to_cell_ids(np.arange(1,n_pts+1))
  filter.set_inputs([seeds[:,0], seeds[:,1], np.zeros(n_pts,dtype='f8'), mesh.areas])
  
  ax = axarr[0,0] #default
  mesh.plot(p_ref,ax,show=False)
  ax.set_xlabel("\n(a) Base field")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  ax = axarr[0,1] #isotrope filter
  filter.filter_radius = 0.1
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  ax.set_xlabel("\n(b) Isotropic filter (R=0.1)")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  ax = axarr[1,0] #anisotrope filter
  filter.filter_radius = [0.4,0.1,1.]
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  ax.set_xlabel("\n(c) Anisotropic filter, R=[0.4,0.1]")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  ax = axarr[1,1] #anisotrope filter with weigthing
  filter.distance_weighting_power = 10
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  ax.set_xlabel("\n(d) Anisotropic filter with a distance weighting power $n=4$")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  plt.show()
  return


def heavyside_filter_fig():
  from HydrOpTop.Filters import Density_Filter, Heavyside_Filter
  import utils
  seeds = np.genfromtxt("seeds.txt")
  group1 = np.genfromtxt("group1.txt",dtype='i8')
  group2 = np.genfromtxt("group2.txt",dtype='i8')
  fig,axarr = plt.subplots(2,2, sharex=True, sharey=True)
  bounding_box = [-0.0001,1.0001,-0.0001,1.0001]
  mesh = utils.voronoi_bounded(seeds,bounding_box)
  n_pts = len(seeds)
  p_ref = np.zeros(len(seeds),dtype='f8')
  p_ref[group1-1] = 1.
  p_ref[group2-1] = 1.
  filter = Density_Filter()
  filter.set_p_to_cell_ids(np.arange(1,n_pts+1))
  filter.set_inputs([seeds[:,0], seeds[:,1], np.zeros(n_pts,dtype='f8'), mesh.areas])
  
  ax = axarr[0,0] #default
  mesh.plot(p_ref,ax,show=False)
  ax.set_xlabel("\n(a) Base field")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  ax = axarr[0,1] #default
  filter.filter_radius = [0.4,0.1,1.]
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  ax.set_xlabel("\n(b) Filtered field with anisotropic density filter\n$R_x=0.4$ and $R_y=0.1$")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  ax = axarr[1,0] 
  H1 = Heavyside_Filter(filter,0.1,5)
  p_til = H1.get_filtered_density(p_bar)
  mesh.plot(p_til,ax,show=False)
  ax.set_xlabel("\n(c) Projected field (Cutoff=0.1, Stepness=5)")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  ax = axarr[1,1] 
  H1 = Heavyside_Filter(filter,0.1,50)
  p_til = H1.get_filtered_density(p_bar)
  mesh.plot(p_til,ax,show=False)
  ax.set_xlabel("\n(d) Projected field (Cutoff=0.1, Stepness=50)")
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  
  return

if __name__ == "__main__":
  density_filter_fig()
  #heavyside_filter_fig()
  plt.show()
