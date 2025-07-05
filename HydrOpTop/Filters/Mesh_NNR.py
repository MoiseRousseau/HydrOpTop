import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, dia_matrix


# Using sparse_distance_matrix
def find_neighbors_within_radius(mesh_centers, ball_radius):
    """
    Given a point cloud `mesh_centers` and a ball radius, return a sparse
    matrix populated with indexes representing the neighbors and the 
    distance as the value.
    This version use the sparse_distance_matrix of the KD tree.
    """
    tree = cKDTree(mesh_centers)
    distance_matrix = tree.sparse_distance_matrix(tree, ball_radius)
    distance_matrix = distance_matrix.tocsr()
    return distance_matrix


# Using query_pairs
def find_neighbors_within_radius2(mesh_centers, ball_radius):
    """
    Given a point cloud `mesh_centers` and a ball radius, return a sparse
    matrix populated with indexes representing the neighbors and the 
    distance as the value.
    This version use the query_pairs of the KD tree.
    Approx 10 times faster than the previous version.
    """
    N = len(mesh_centers)
    tree = cKDTree(mesh_centers)
    pairs = tree.query_pairs(ball_radius, output_type="ndarray")
    distance = np.sqrt(
        np.sum((mesh_centers[pairs[:,0]] - mesh_centers[pairs[:,1]])**2,axis=1)
    )
    distance_matrix = csr_matrix((distance, pairs.T), shape=(N,N))
    distance_matrix += distance_matrix.T
    # explicit 0 along the diagonal
    distance_matrix += dia_matrix((np.zeros(N)+1e-40,0), shape=(N,N))
    return distance_matrix


#test
if __name__ == "__main__":
    import time
    n_pts = 200000
    np.random.seed(1234)
    points = np.random.rand(n_pts, 3) * 1000
    ball_radius = 40

    start = time.time()
    distance_matrix = find_neighbors_within_radius(points,ball_radius)
    print(time.time() - start)

    start = time.time()
    distance_matrix2 = find_neighbors_within_radius2(points,ball_radius)
    print(time.time() - start)

