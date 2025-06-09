import logging
import time
from functools import partial

# Mathematical functions
import numpy as np
import scipy.sparse as scsp
import scipy.sparse.linalg as scspl


def buildLaplacianMatrix(n_time, stepsize_time, n_vertices, area_vertices, laplacian_space, eps=0.0):
    """Return a function which inverts the space-time Laplacian
    """
    # Laplacian matrix in Time
    laplacian_time = np.zeros((n_time + 1, n_time + 1))

    for alpha in range(1, n_time):
        laplacian_time[alpha, alpha] = -2.0
        laplacian_time[alpha, alpha + 1] = 1.0
        laplacian_time[alpha, alpha - 1] = 1.0

    laplacian_time[0, 1] = 1.0
    laplacian_time[0, 0] = -1.0
    laplacian_time[-1, -2] = 1.0
    laplacian_time[-1, -1] = -1.0
    laplacian_time *= 1 / (stepsize_time ** 2)

    # Diagonalizing in Time and factorizing in D
    startFact = time.time()

    eigen_val_time, eigen_vect_time = np.linalg.eigh(laplacian_time)
    list_factor = []

    for alpha in range(n_time + 1):
        factor = scspl.factorized(
            (
                laplacian_space
                + (eigen_val_time[alpha] - eps) * scsp.diags([area_vertices], [0])
            ).tocsc()
        )
        list_factor.append(factor)

    endFact = time.time()

    logging.debug(
        "---- Laplace matrix ".ljust(42, '-') + "\n"
        "Factorizing the Laplace matrix: " + str(round(endFact - startFact, 2)) + "s."
    )

    return partial(__laplacian_invert, n_time, n_vertices, eigen_vect_time, list_factor)

def __laplacian_invert(n_time, n_vertices, eigen_vect_time, list_factor, input):
    # Diagonalizing
    input_diag = np.array(np.dot(eigen_vect_time.transpose(), input))

    # Solving for each line eigenvector
    solution = np.zeros((n_time + 1, n_vertices))
    for alpha in range(n_time + 1):
        solution[alpha, :] = list_factor[alpha](input_diag[alpha, :])

    return np.array(np.dot(eigen_vect_time, solution))

