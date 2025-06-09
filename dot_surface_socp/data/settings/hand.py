from numpy import zeros as np_zeros
from numpy import where as np_where


def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    for i in range(n_vertices):
        mub0[i] += area_vertices[i] * np_where(vertices[i, 1] < -0.5, 1, 0)
        mub1[i] += area_vertices[i] * np_where(vertices[i, 1] > 0.4, 1, 0)

    return mub0, mub1
