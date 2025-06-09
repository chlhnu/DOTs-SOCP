from numpy import zeros as np_zeros
from dot_surface_socp.data.util import cut_off

def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    for i in range(n_vertices):
        mub0[i] += (
            area_vertices[i]
            * cut_off(vertices[i, 0] + 0.357,0.007)
            * cut_off(vertices[i, 1] + 0.9,0.1)
            * cut_off(-vertices[i, 2] + 0.02748,0.00422)
        )
        mub1[i] += (
            area_vertices[i]
            * cut_off(-vertices[i, 0] + 0.715, 0.0143)
            * cut_off(vertices[i, 1] + 0.9, 0.1)
            * cut_off(vertices[i, 2] + 0.2389, 0.02114)
        )
        mub1[i] += (
            area_vertices[i]
            * cut_off(-vertices[i, 0] + 0.715, 0.0143)
            * cut_off(vertices[i, 1] + 0.9, 0.1)
            * cut_off(-vertices[i, 2] + 0.3023, 0.02114)
        )
        mub1[i] += (
            area_vertices[i]
            * cut_off(-vertices[i, 0] + 0.286, 0.0143)
            * cut_off(vertices[i, 1] + 0.9, 0.1)
            * cut_off(vertices[i, 2] + 1.0844, 0.02114)
        )

    return mub0, mub1
