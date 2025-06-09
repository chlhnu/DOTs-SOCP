from numpy import zeros as np_zeros
from dot_surface_socp.data.util import cut_off

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    for i in range(n_vertices):
        mub0[i] = (
            area_vertices[i]
            * cut_off(vertices[i, 0] + 0.2626, 0.01)
            * cut_off(vertices[i, 1] + 0.9108, 0.1012)
        )
        mub1[i] = (
            area_vertices[i]
            * cut_off(- vertices[i, 0] + 0.9696, 0.0202)
            * cut_off(vertices[i, 1] + 0.9108, 0.1012)
            * cut_off(vertices[i, 2] + 0.3371, 0.0337)
        )
        mub1[i] += (
            area_vertices[i]
            * cut_off(- vertices[i, 0] + 0.9696, 0.0202)
            * cut_off(vertices[i, 1] + 0.9108, 0.1012)
            * cut_off(vertices[i, 2] + 0.4383, 0.0337)
        )

    return mub0, mub1
