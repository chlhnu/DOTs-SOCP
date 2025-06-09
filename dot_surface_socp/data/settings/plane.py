from numpy import zeros as np_zeros
from numpy import array as np_array
from dot_surface_socp.data.util import gaussian

# Parameters of mu0 on x-y plane
CENTER0 = np_array([0.4, 0.4, 0.0])
SCALE0 = 2 * (0.1 ** 2)

# Parameters of mu1 on x-y plane
CENTER1 = np_array([0.6, 0.6, 0.0])
SCALE1 = 2 * (0.1 ** 2)


def get_mu(area_vertices, vertices):
    n_vertices = vertices.shape[0]

    mub0 = np_zeros(n_vertices)
    mub1 = np_zeros(n_vertices)

    for i in range(n_vertices):
        vertex = vertices[i, :]
        area_vertex = area_vertices[i]
        mub0[i] = area_vertex * gaussian(vertex, CENTER0, SCALE0)
        mub1[i] = area_vertex * gaussian(vertex, CENTER1, SCALE1)

    return mub0, mub1


def get_exact_transportation(t_array, vertices, area_vertices):
    _scale0_quarter_power = SCALE0 ** 0.25
    _scale1_quarter_power = SCALE1 ** 0.25
    get_sigma_t = lambda t: ((1 - t) * _scale0_quarter_power + t * _scale1_quarter_power) ** 4
    get_center_t = lambda t: (1 - t) * CENTER0 + t * CENTER1

    n_vertices = vertices.shape[0]
    mu = np_zeros((t_array.shape[0], n_vertices))
    for id_t, t in enumerate(t_array):
        sigma_t = get_sigma_t(t)
        center_t = get_center_t(t)

        for id_vertex in range(n_vertices):
            vertex = vertices[id_vertex, :]
            area_vertex = area_vertices[id_vertex]
            mu[id_t, id_vertex] = area_vertex * gaussian(vertex, center_t, sigma_t)
    
    return mu
