from numpy import zeros as np_zeros
from numpy import exp as np_exp
from numpy import linalg as lin
from dot_surface_socp.data.util import cut_off

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    length_scale = 0.1
    center0 = vertices[4492, :]
    center1 = vertices[4225, :]

    for i in range(n_vertices):
        alpha = 0.1 * vertices[i, 0] + vertices[i, 1]
        beta = - vertices[i, 0] + 0.1 * vertices[i, 1]
        gamma = vertices[i, 2]

        if gamma >= -0.1:
            mub0[i] = area_vertices[i] * cut_off(- 0.2 - alpha, 0.3) * cut_off(alpha - 0.15, 0.3) * cut_off(
                0.1 - beta, 0.3) * cut_off(beta - 0.45, 0.3)

        mub1[i] += area_vertices[i] * np_exp(-lin.norm(vertices[i, :] - center0) ** 2 / length_scale ** 2)
        mub1[i] += area_vertices[i] * np_exp(-lin.norm(vertices[i, :] - center1) ** 2 / length_scale ** 2)

    return mub0, mub1
