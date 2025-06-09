import numpy as np
from numpy.linalg import norm as np_norm
from dot_surface_socp.data.util import gaussian, cut

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np.zeros(n_vertices)
    mub1 = np.zeros(n_vertices)

    center0 = vertices[2786, :]
    center1 = vertices[1232, :]
    center2 = vertices[406, :]

    for i in range(n_vertices):
        vertice = vertices[i]
        area = area_vertices[i]
        mub0[i] += area * cut(np_norm(vertice - center0) < 0.5, gaussian(vertice, center0, 0.5))
        mub1[i] += area * cut(np_norm(vertice - center1) < 0.5, gaussian(vertice, center1, 0.5))
        mub1[i] += area * cut(np_norm(vertice - center2) < 0.5, gaussian(vertice, center2, 0.5))

    return mub0, mub1
