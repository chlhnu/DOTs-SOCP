from numpy import zeros as np_zeros
from numpy import where as np_where
from dot_surface_socp.data.util import cut_off

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)


    for i in range(n_vertices):
        mub0[i] += area_vertices[i] * np_where(vertices[i, 0] > 0.03, 1, 0)
        mub1[i] += area_vertices[i] * cut_off(-vertices[i, 1] + 0.3, 0.5) * np_where(vertices[i, 0] < -0.06, 1, 0) * np_where(vertices[i, 1] < 0.11, 1, 0)* np_where(vertices[i, 1] > 0.05, 1, 0)

    return mub0, mub1
