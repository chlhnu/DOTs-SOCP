import numpy as np
from numpy.linalg import norm as np_norm
from dot_surface_socp.data.util import gaussian, cut

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np.zeros(n_vertices)
    mub1 = np.zeros(n_vertices)

    center0 = np.array([0.0888, 1.282, 0.512])
    center11 = np.array([-1.035, -1.087, 0.300])
    center12 = np.array([1.212, -0.594, 0.455])

    for i in range(n_vertices):
        vertice = vertices[i]
        area = area_vertices[i]
        mub0[i] += area * cut(np_norm(vertice - center0) < 0.5, gaussian(vertice, center0, 0.3))
        mub1[i] += area * cut(np_norm(vertice - center11) < 0.3, gaussian(vertice, center11, 0.3))
        mub1[i] += area * cut(np_norm(vertice - center12) < 0.3, gaussian(vertice, center12, 0.3))

    return mub0, mub1
