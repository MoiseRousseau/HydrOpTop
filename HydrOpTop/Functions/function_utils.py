import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator

def df_dX(interpolator, point):
    """
    Compute derivative of LinearNDInterpolator or NearestNDInterpolator output
    w.r.t. X values at the given point.
    """
    # LINEAR case
    if isinstance(interpolator, LinearNDInterpolator):
        tri = interpolator.tri
        simp_idx = tri.find_simplex(point)
        if simp_idx == -1:
            raise ValueError("Point is outside the convex hull.")
        
        vertices = tri.simplices[simp_idx]
        T = tri.transform[simp_idx, :tri.ndim]
        r = tri.transform[simp_idx, tri.ndim]
        bary = np.dot(T, point - r)
        bary = np.append(bary, 1 - bary.sum())

        df = np.zeros(tri.npoints)
        df[vertices] = bary
        return df

    # NEAREST case
    elif isinstance(interpolator, NearestNDInterpolator):
        # Cache KDTree if not already done
        if not hasattr(interpolator, "_kdtree"):
            interpolator._kdtree = KDTree(interpolator.points)

        _, idx = interpolator.tree.query(point, k=1)
        df = np.zeros(len(interpolator.points))
        df[idx] = 1.0
        return df
    
    # RBF
    # TODO: test
    elif isinstance(interpolator, RBFInterpolator):
        y = np.atleast_2d(point)

        if False:
            # Build kernel matrix A (same used in fit)
            if not hasattr(interpolator, "_Ainv"):
                coords = interpolator.yi
                kernel = interpolator.kernel
                eps = interpolator.epsilon

                # pairwise distances
                r = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
                A = interpolator._h_function(r)
                if eps is not None and eps != 0:
                    A = interpolator._h_function(r / eps)
                interpolator._Ainv = np.linalg.inv(A)

        Ainv = interpolator._Ainv
        coords = interpolator.yi
        eps = interpolator.epsilon

        # Compute Phi(p, coords)
        r_p = np.linalg.norm(coords - y, axis=1)
        if eps is not None and eps != 0:
            r_p = r_p / eps
        Phi = interpolator._h_function(r_p)

        # Derivative
        df_dX = Phi @ Ainv
        return df_dX.ravel()

    else:
        raise TypeError("Interpolator must be LinearNDInterpolator, NearestNDInterpolator or RBFInterpolator.")
