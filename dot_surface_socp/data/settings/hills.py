from numpy import zeros as np_zeros
from dot_surface_socp.data.util import gaussian

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    center0 = vertices[1191, :]
    center1 = vertices[9505, :]

    for i in range(n_vertices):
        vertice, area = vertices[i], area_vertices[i]
        mub0[i] = area * gaussian(vertice, center0, 1.0)
        mub1[i] = area * gaussian(vertice, center1, 1.0)

    return mub0, mub1
