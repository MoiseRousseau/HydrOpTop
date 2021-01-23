#
# Verification of the sensitivity of the filtered density and derivative calculation
#

import sys
import os
path = os.getcwd() + '/../../'
sys.path.append(path)

import numpy as np

from HydrOpTop.Filters import Density_Filter
import utils





if __name__ == "__main__":
  #create filter instance
  filter = Density_Filter(0.2)
  #sample random point
  n_pts = 500
  points = np.random.rand(n_pts,2)
  bounding_box = [0.,1.,0.,1.]
  mesh = utils.voronoi_bounded(points,bounding_box)
  #update filter
  filter.set_p_to_cell_ids(np.arange(1,n_pts+1))
  filter.set_inputs([points[:,0], points[:,1], np.zeros(n_pts,dtype='f8'), mesh.areas])
  
  #compare initial p and filtered p_bar
  p = points[:,1] - (10*(points[:,0]-0.5)**2+0.2)
  p[p>0] = 1.
  p[p<=0.] = 0.01
  #mesh.plot(p, show=False)
  #filtering
  p_bar = filter.get_filtered_density(p)
  #mesh.plot(p_bar)
  
  #derivative
  deriv = filter.get_filter_derivative(p)
  
  #compare with finite difference
  i = 0 #derivative of the filtered density wrt p[0]
  # compute FD
  old_p = p[i]
  pertub = old_p * 1e-3
  p[i] = old_p + pertub
  p_bar_pertub = filter.get_filtered_density(p)
  deriv_fd = (p_bar_pertub - p_bar) / pertub
  
  #compare both
  error = False
  print(f"Compare row {i}")
  row = deriv.getrow(i)
  row_indices, row_data = row.indices, row.data
  print("Row\tColumn\tDerivative")
  for count,j in enumerate(row_indices):
    fd = deriv_fd[j]
    analytic = row_data[count]
    print(f"{i}\t{j}\t{analytic:.6e} vs {fd:.6e}\t",end="")
    if abs(1-fd/analytic) < 1e-3: print("OK")
    else: 
      print("X")
      error = True
  
  print("\n")
  if error: exit(1)
  else: exit(0)
