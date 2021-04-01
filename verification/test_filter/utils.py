#from https://gist.github.com/pv/8036995

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull

class voronoi_bounded:
    def __init__(self,towers, bounding_box):
        eps = 1e-6
        # Select towers inside the bounding box
        i = np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                                 towers[:, 0] <= bounding_box[1]),
                                  np.logical_and(bounding_box[2] <= towers[:, 1],
                                                 towers[:, 1] <= bounding_box[3]))
        # Mirror points
        points_center = towers[i, :]
        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
        points = np.append(points_center,
                           np.append(np.append(points_left,
                                               points_right,
                                               axis=0),
                                     np.append(points_down,
                                               points_up,
                                               axis=0),
                                     axis=0),
                           axis=0)
        # Compute Voronoi
        vor = Voronoi(points)
        # Filter regions
        regions = []
        point_region = []
        count = 0
        for index in vor.point_region:
            region = vor.regions[index]
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = vor.vertices[index, 0]
                    y = vor.vertices[index, 1]
                    if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                           bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                        flag = False
                        break
            if region != [] and flag:
                regions.append(region)
                point_region.append(count)
                count += 1
        vor.filtered_points = points_center
        vor.filtered_regions = regions
        vor.filtered_point_region = point_region
        
        self.vor = vor
        self.areas = np.zeros(len(self.vor.filtered_regions), dtype='f8')
        for i,region in enumerate(self.vor.filtered_regions):
            polygon = self.vor.vertices[region]
            convexhull = ConvexHull(polygon)
            self.areas[i] = convexhull.area
            
        return
        

    def plot(self, color=None, show=True, title=""):
        if color is None:
            color = self.areas
        
        rgb = (color - np.min(color)) / (np.max(color) - np.min(color))
        rgb = [[0,x,0] for x in rgb]
        
        fig = plt.figure()
        ax = fig.gca()
        # Plot initial points
        ax.plot(self.vor.filtered_points[:, 0], self.vor.filtered_points[:, 1], 'b.')
        # Plot ridges
        for region in self.vor.filtered_regions:
            vertices = self.vor.vertices[region + [region[0]], :]
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
            
        #fill polygon
        for i in range(len(color)):
            j = self.vor.filtered_point_region[i]
            region = self.vor.filtered_regions[j]
            polygon = self.vor.vertices[region]
            plt.fill(*zip(*polygon), alpha=0.4, c=rgb[i])
        
        ax.set_title(title)
        if show:
          plt.show()
        return




if __name__ == "__main__":
    points = np.random.rand(100,2)
    obj = voronoi_bounded(points, [0,1, 0,1])
    obj.plot()
