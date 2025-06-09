from numpy import zeros as np_zeros
from numpy import exp as np_exp
from numpy import linalg as lin
from dot_surface_socp.data.util import cut_off
from numpy import where as np_where

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)
    length_scale = 0.1

    center0 = vertices[5982, :]
    center1 = vertices[1347, :]

    for i in range(n_vertices):
        mub0[i] += area_vertices[i] * np_exp(-lin.norm(vertices[i, :] - center0) ** 2 / length_scale ** 2)
        mub0[i] += area_vertices[i] * np_exp(-lin.norm(vertices[i, :] - center1) ** 2 / length_scale ** 2)
        mub1[i] += area_vertices[i] * np_where(vertices[i, 1] > 0.4, 1, 0)

    return mub0, mub1
