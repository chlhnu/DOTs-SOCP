import numpy as np
import logging
from dot_surface_socp.config import LOG_LEVELS
from dot_surface_socp.utils.type import ErrorVersusExactData
from dot_surface_socp.utils.util import translate_density, l1_norm, l2_norm, linf_norm

def check_mass_conservation(mu, verbose=True):
    mass_time_layers = np.array([np.sum(mu[idx, :], axis=0) for idx in range(mu.shape[0])])
    error_mass_conservation = np.linalg.norm(mass_time_layers - 1.0) / np.sqrt(mass_time_layers.shape[0])

    if verbose:
        msg_summary = f"Mass Conservation Violation: {error_mass_conservation:.2e}"
        with np.printoptions(precision=4, suppress=True):
            msg_time_layer = f"Sum of Mass at each time layer:\n{mass_time_layers}"

        logging.log(LOG_LEVELS["info"],
            "---- Mass Conservation ".ljust(42, '-') + "\n"
            f"{msg_time_layer}\n"
            f"{msg_summary}"
        )

    return error_mass_conservation

def check_negative_mass(mu, verbose=True):
    n_time = mu.shape[0]
    negative_mass_time_layers = np.zeros(n_time)

    for idx in range(n_time):
        rho_t = mu[idx, :]
        negative_mass_time_layers[idx] = np.add.reduce(rho_t, where=rho_t < 0)

    error_non_negative_mass = np.linalg.norm(negative_mass_time_layers) / np.sqrt(negative_mass_time_layers.shape[0])

    if verbose:
        msg_summary = f"Non-Negative Mass Violation: {error_non_negative_mass:.2e}"
        with np.printoptions(precision=4, suppress=True):
            msg_time_layer = f"Sum of Negative Mass at each time layer:\n{negative_mass_time_layers}"

        logging.log(LOG_LEVELS["info"],
            "---- Negative Mass ".ljust(42, '-') + "\n"
            f"{msg_time_layer}\n"
            f"{msg_summary}"
        )

    return error_non_negative_mass, negative_mass_time_layers

def compare_with_exact_transportation(mu, mu_exact, geometry, verbose=True):
    _mu = translate_density(mu, geometry)
    _mu_exact = translate_density(mu_exact, geometry)
    diff = _mu - _mu_exact
    mesh_area = geometry["area_vertices"][np.newaxis, :] / 3.0

    error_transportation = ErrorVersusExactData(
        l1=l1_norm(diff, weight=mesh_area) / (1.0 + l1_norm(_mu_exact, weight=mesh_area)),
        l2=l2_norm(diff, weight=mesh_area) / (1.0 + l2_norm(_mu_exact, weight=mesh_area)),
        linf=linf_norm(diff) / (1.0 + linf_norm(_mu_exact)),
    )

    if verbose:
        msg_summary = f"L_1 Error: {error_transportation["l1"]:.2e}\n"\
                      f"L_2 Error: {error_transportation["l2"]:.2e}\n"\
                      f"L_Inf Error: {error_transportation["linf"]:.2e}"

        logging.log(LOG_LEVELS["info"],
            "---- Versus exact transportation ".ljust(42, '-') + "\n"
            f"{msg_summary}"
        )

    return error_transportation
