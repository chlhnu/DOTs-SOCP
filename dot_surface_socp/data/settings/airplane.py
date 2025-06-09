from numpy import zeros as np_zeros
from dot_surface_socp.data.util import cut_off

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    for i in range(n_vertices):
        mub0[i] = area_vertices[i] * cut_off(- (vertices[i, 2] - 0.5), 0.3)
        mub1[i] = area_vertices[i] * cut_off(vertices[i, 2] + 0.1, 0.3)

    return mub0, mub1
