# Uses Taylor expansion to make expm faster
import scipy.linalg as slin
import scipy.sparse.linalg as sslin
import numpy as np
import glog as log
from scipy.sparse import csc_matrix


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
        A = expm0.copy()
        B = csc_matrix(W1 - W0)
        expm_term = sslin.expm(B/2).todense()
        new_expm = (expm_term.dot(A)).dot(expm_term)
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
