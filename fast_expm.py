# Uses Taylor expansion to make expm faster
import scipy.linalg as slin
import numpy as np


def fast_expm(W1, W0=None, expm0=None,
              threshold=1e-5, num_terms=20, debug=False):
    if W0 is None and expm0 is None:
        return(slin.expm(W1))
    elif (W0 is None and expm0 is not None) or (W0 is not None and expm0 is None):
        print("Please specify both original matrix and original exponential if fast exponentiation is wanted.")
        return(slin.expm(W1))
    elif np.max(np.abs(W0 - W1)) >= threshold:
        return(slin.expm(W1))
    else:
        # Uses Taylor expansion
        new_expm = expm0
        factorial_term = 1
        matrix_term = W1 - W0
        diff = W1 - W0
        for i in range(1, num_terms + 1):
            new_expm += matrix_term / factorial_term
            factorial_term *= i
            matrix_term *= diff
            if debug:
                print(i)
        if debug:
            true_result = slin.expm(W1)
            print(true_result)
            print(np.linalg.norm(new_expm - true_result))
        return new_expm
