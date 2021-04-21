import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../')
sys.path.append('../../../verification/test_filter/')

def density_filter_fig():
  from HydrOpTop.Filters import Density_Filter
  import utils
  seeds = np.genfromtxt("seeds.txt")
  group1 = np.genfromtxt("group1.txt",dtype='i8')
  group2 = np.genfromtxt("group2.txt",dtype='i8')
  fig,axarr = plt.subplots(2,2)
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
  ax.set_xlabel("Reference field")
  
  ax = axarr[0,1] #isotrope filter
  filter.filter_radius = 0.1
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  ax.set_xlabel("Isotropic filter (R=0.1)")
  
  ax = axarr[1,0] #anisotrope filter
  filter.filter_radius = [0.4,0.1,1.]
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  ax.set_xlabel("Anisotropic filter (R=[0.3,0.1])")
  
  ax = axarr[1,1] #anisotrope filter with weigthing
  filter.distance_weighting_power = 4
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  
  plt.show()



def heavyside_filter_fig():
  from HydrOpTop.Filters import Density_Filter, Heavyside_Density_Filter
  import utils
  seeds = np.genfromtxt("seeds.txt")
  group1 = np.genfromtxt("group1.txt",dtype='i8')
  group2 = np.genfromtxt("group2.txt",dtype='i8')
  fig,axarr = plt.subplots(2,2)
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
  filter.filter_radius = [0.4,0.1,1.]
  filter.initialize()
  p_bar = filter.get_filtered_density(p_ref)
  mesh.plot(p_bar,ax,show=False)
  ax.set_xlabel("Base field (filtered with anisotropic density filter)")
  
  ax = axarr[0,1] 
  H1 = Heavyside_Density_Filter(filter,0.5,2)
  p_til = H1.get_filtered_density(p_bar)
  mesh.plot(p_til,ax,show=False)
  ax.set_xlabel("Cutoff=0.5, Stepness=2")
  
  ax = axarr[1,0] 
  H1 = Heavyside_Density_Filter(filter,0.5,10)
  p_til = H1.get_filtered_density(p_bar)
  mesh.plot(p_til,ax,show=False)
  ax.set_xlabel("Cutoff=0.5, Stepness=10")
  
  ax = axarr[1,1] 
  H1 = Heavyside_Density_Filter(filter,0.2,10)
  p_til = H1.get_filtered_density(p_bar)
  mesh.plot(p_til,ax,show=False)
  ax.set_xlabel("Cutoff=0.2, Stepness=10")
  return

if __name__ == "__main__":
  #density_filter_fig()
  heavyside_filter_fig()
  plt.show()
