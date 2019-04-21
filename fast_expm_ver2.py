# Uses Taylor expansion to make expm faster
import scipy.linalg as slin
import numpy as np
import glog as log

def _smart_matrix_product(A, B, alpha=None, structure=None):
    """
    A matrix product that knows about sparse and structured matrices.
    Parameters
    ----------
    A : 2d ndarray
        First matrix.
    B : 2d ndarray
        Second matrix.
    alpha : float
        The matrix product will be scaled by this constant.
    structure : str, optional
        A string describing the structure of both matrices `A` and `B`.
        Only `upper_triangular` is currently supported.
    Returns
    -------
    M : 2d ndarray
        Matrix product of A and B.
    """
    if len(A.shape) != 2:
        raise ValueError('expected A to be a rectangular matrix')
    if len(B.shape) != 2:
        raise ValueError('expected B to be a rectangular matrix')
    f = None
    if f is not None:
        if alpha is None:
            alpha = 1.
        out = f(alpha, A, B)
    else:
        if alpha is None:
            out = A.dot(B)
        else:
            out = alpha * A.dot(B)
    return out

def crit(W1, W0, threshold=1e-5, norm='l2'):
    if norm == 'l1':
        return np.linalg.norm(W0 - W1, ord=1)
    if norm == 'l2':
        return np.linalg.norm(W0 - W1) <= threshold
    elif norm == 'linf':
        return np.max(np.abs(W0 - W1)) <= threshold
    elif norm == 'always_true':
        return True

def fast_expm(W1, W0=None, expm0=None,
              threshold=1e-5, num_terms=15, debug=True,
              crit_norm='l2'):
    if W0 is None and expm0 is None:
        return(slin.expm(W1))
    elif (W0 is None and expm0 is not None) or (W0 is not None and expm0 is None):
        log.info("Please specify both original matrix and original exponential if fast exponentiation is wanted.")
        return(slin.expm(W1))
    elif not crit(W0, W1, threshold=threshold, norm=crit_norm):
        # If the criterion is not satisfied
        return(slin.expm(W1))
    else:
        log.info("Fast exponentiation started")
        new_expm = expm0.copy()
        factorial_term = 1
        matrix_term1 = W1.copy()
        matrix_term2 = W0.copy()
        for i in range(1, num_terms + 1):
            factorial_term *= i
            new_expm += (matrix_term1 - matrix_term2)/factorial_term
            matrix_term1 = np.dot(matrix_term1, W1)
            matrix_term2 = np.dot(matrix_term2, W0)
        if debug:
            log.info("Fast exponentiation finished, expm started")
            true_result = slin.expm(W1)
            log.info("Expm finished")

            log.info("The l2 norm of the difference is {}".format(np.linalg.norm(new_expm - true_result)))
            log.info("The proportion of error wrt l2 norm is {}".format(
                np.linalg.norm(new_expm - true_result)/np.linalg.norm(true_result)
            ))
            log.info("The l1 norm of the difference is {}".format(np.linalg.norm(new_expm - true_result, ord=1)))
            log.info("The proportion of error wrt l1 norm is {}".format(
                np.linalg.norm(new_expm - true_result, ord=1)/np.linalg.norm(true_result, ord=1)
            ))
            log.info("The linf norm of the difference is {}".format(np.linalg.norm(new_expm - true_result, ord=np.inf)))
            log.info("The proportion of error wrt linf norm is {}".format(
                np.linalg.norm(new_expm - true_result, ord=np.inf)/np.linalg.norm(true_result, ord=np.inf)
            ))

        return new_expm
