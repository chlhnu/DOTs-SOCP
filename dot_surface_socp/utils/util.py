"""Util functions
"""

from dot_surface_socp.utils.type import GeometryData
import numpy as np

def translate_density(mu: np.ndarray, geometry: GeometryData, reverse = False) -> np.ndarray:
    """Translate density with multiplying or dividing by mesh area

    Args:
        mu (ndarray):
            mu, shape (N,) or (ntime, N)
        geometry (GeometryData):
            geometry data
        reverse (bool, optional):
            if True, multiply by area;
            if False, divide by area. (Default)

    Returns:
        ndarray: translated density with same shape as input
    """
    if mu.ndim == 1:
        mesh_area = geometry["area_vertices"] / 3.0
    else:
        mesh_area = geometry["area_vertices"][np.newaxis, :] / 3.0

    if reverse:
        return np.multiply(mu, mesh_area)
    else:
        return np.divide(mu, mesh_area)

def l1_norm(v: np.ndarray, weight: np.ndarray = None) -> float:
    """Compute the l1 norm ||v||_{weight,1}
    """
    if v.ndim == 1:
        if weight is not None:
            return float(np.sum(np.multiply(np.abs(v), weight)))
        else:
            return float(np.sum(np.abs(v)))
    elif v.ndim == 2:
        time_stepsize = 1.0 / v.shape[0]
        if weight is not None:
            return float(np.sum(np.multiply(np.abs(v), weight)) * time_stepsize)
        else:
            return float(np.sum(np.abs(v)) * time_stepsize)
    else:
        raise NotImplementedError("Only support 1D and 2D")

def l2_norm(v: np.ndarray, weight: np.ndarray = None) -> float:
    """Compute the l2 norm ||v||_{weight,2}
    """
    if v.ndim == 1:
        if weight is not None:
            return float(np.sqrt(np.sum(np.multiply(np.square(v), weight))))
        else:
            return float(np.sqrt(np.sum(np.square(v))))
    elif v.ndim == 2:
        time_stepsize = 1.0 / v.shape[0]
        if weight is not None:
            return float(np.sqrt(np.sum(np.multiply(np.square(v), weight)) * time_stepsize))
        else:
            return float(np.sqrt(np.sum(np.square(v)) * time_stepsize))
    else:
        raise NotImplementedError("Only support 1D and 2D")

def linf_norm(v: np.ndarray) -> float:
    """Compute the linf norm ||v||_{inf}
    """
    return float(np.max(np.abs(v)))