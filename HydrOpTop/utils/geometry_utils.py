import numpy as np

def element_center(points):
    """
    Compute the centroid of a polyhedron or polygon
    points: (N, M) array of vertex coordinates
    Returns: (M,) centroid coordinate
    """
    return np.mean(points, axis=0)

def element_area(points):
    """Area of a 2D polygon using the shoelace formula"""
    if len(points) == 3:
    	res = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    else:
        x, y = zip(*points)
        res = 0.5 * abs(sum(x[i] * y[(i+1)%len(points)] - x[(i+1)%len(points)] * y[i] for i in range(len(points))))
    return res

def element_volume(pts):
    if len(pts) == 4:
        return tetrahedron_volume(pts)
    elif len(pts) == 5:
        return pyramid_volume(pts)
    elif len(pts) == 6:
        return wedge_volume(pts)
    elif len(pts) == 8:
        return hexahedron_volume(pts)


def tetrahedron_volume(pts):
    """Volume of a tetrahedron"""
    a, b, c, d = pts
    return abs(np.dot((a - d), np.cross((b - d), (c - d)))) / 6

def hexahedron_volume(pts):
    """
    Estimate volume of a hexahedron by dividing into tets
    pts: (8, 3) array of the 8 corner points in standard hexahedron order
    """
    # Split into 5 tets
    a, b, c, d, e, f, g, h = pts
    tets = [
        [a, b, d, e],
        [b, c, d, f],
        [d, e, f, h],
        [b, d, f, e],
        [b, f, e, g],
    ]
    return sum(tetrahedron_volume(tet) for tet in tets)

def pyramid_volume(base_pts, apex):
    """Volume of a pyramid given base polygon and apex point"""
    base_centroid = np.mean(base_pts, axis=0)
    base_area = polygon_area_2d([(p[0], p[1]) for p in base_pts])  # assumes base is flat on XY
    height = abs(apex[2] - base_centroid[2])
    return (1/3) * base_area * height

def wedge_volume(pts):
    """
    Volume of a wedge (triangular prism) defined by 6 points:
    - Base triangle: a, b, c
    - Top triangle: d, e, f
    Assumes point order (a→d, b→e, c→f are vertical edges)
    """
    # Split into 3 tetrahedra
    a, b, c, d, e, f = pts
    vol = (
        tetrahedron_volume([a, b, c, d]) +
        tetrahedron_volume([b, c, d, e]) +
        tetrahedron_volume([c, d, e, f])
    )
    return vol
