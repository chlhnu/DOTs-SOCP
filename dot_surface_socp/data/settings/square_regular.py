from numpy import zeros as np_zeros
from numpy import array as np_array
from numpy import linalg as lin
from dot_surface_socp.data.util import cut_off

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    x0 = np_array([0.33, 0.5, 0.0])
    x10 = np_array([0.8, 0.2, 0.0])
    x11 = np_array([0.8, 0.8, 0.0])

    for i in range(n_vertices):
        mub0[i] = area_vertices[i] * cut_off(lin.norm(vertices[i, :] - x0) - 0.1, 0.1)
        mub1[i] += area_vertices[i] * cut_off((lin.norm(vertices[i, :] - x10) - 0.1) * 2., 0.1)
        mub1[i] += area_vertices[i] * cut_off((lin.norm(vertices[i, :] - x11) - 0.1) * 2., 0.1)

    return mub0, mub1
