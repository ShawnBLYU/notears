"""Implementation of the simple 50-line version of NOTEARS algorithm.

Defines the h function, the augmented Lagrangian, and its gradient.
Each augmented Lagrangian subproblem is minimized by L-BFGS-B from scipy.

Note: this version implements NOTEARS without l1 regularization,
i.e. lambda = 0, hence it requires n >> d.

Uses the alternative form for h for profiling
"""
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import time
import glog as log

    
def notears_simple(X: np.ndarray,
                   max_iter: int = 100,
                   h_tol: float = 1e-8,
                   w_threshold: float = 0.3) -> np.ndarray:
    """Solve min_W ell(W; X) s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X: [n,d] sample matrix
        max_iter: max number of dual ascent steps
        h_tol: exit if |h(w)| <= h_tol
        w_threshold: fixed threshold for edge weights

    Returns:
        W_est: [d,d] estimate
    """
    def _h(w):
        start = time.time()
        W = w.reshape([d, d])
        result = np.trace(np.linalg.matrix_power(np.eye(d) + constraint_alpha * W, d))
        end = time.time()
        return result

    def _func(w):
        W = w.reshape([d, d])
        start1 = time.time()
        loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))
        end1 = time.time()
        log.info("time1 = {}".format(end1 - start1))
        start2 = time.time()
        h = _h(W)
        end2 = time.time()
        log.info("time2 = {}".format(end2 - start2))
        result = loss + 0.5 * rho * h * h + alpha * h
        return result

    def _grad(w):
        W = w.reshape([d, d])
        start1 = time.time()
        loss_grad = - 1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W)
        end1 = time.time()
        log.info("time1 = {}".format(end1 - start1))

        start2 = time.time()
        E = slin.expm(W * W)
        constraint_grad = (rho * (np.trace(E) - d) + alpha) * E.T * W * 2
        end2 = time.time()
        log.info("time2 = {}".format(end2 - start2))

        obj_grad = loss_grad + constraint_grad
        result = obj_grad.flatten()
        return result

    n, d = X.shape
    constraint_alpha = 0.01
    w_est, w_new = np.zeros(d * d), np.zeros(d * d)
    rho, alpha, h, h_new = 1.0, 0.0, np.inf, np.inf
    bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        while rho < 1e+20:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=_grad, bounds=bnds)
            w_new = sol.x
            h_new = _h(w_new)
            if h_new > 0.25 * h:
                log.info("rejected")
                rho *= 10 #REJECT
            else:
                log.info("accepted")
                break #ACCEPT
        log.info("change = {}".format(np.linalg.norm(w_est - w_new)))
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol:
            break
    w_est[np.abs(w_est) < w_threshold] = 0
    return w_est.reshape([d, d])


if __name__ == '__main__':
    import networkx as nx
    import utils

    # configurations
    n, d = 1000, 40
    graph_type, degree, sem_type = 'erdos-renyi', 2, 'linear-gauss'
    log.info('Graph: %d node, avg degree %d, %s graph', d, degree, graph_type)
    log.info('Data: %d samples, %s SEM', n, sem_type)

    # graph
    log.info('Simulating graph ...')
    G = utils.simulate_random_dag(d, degree, graph_type)
    log.info('Simulating graph ... Done')

    # data
    log.info('Simulating data ...')
    X = utils.simulate_sem(G, n, sem_type)
    log.info('Simulating data ... Done')

    # solve optimization problem
    log.info('Solving equality constrained problem ...')
    W_est = notears_simple(X)
    G_est = nx.DiGraph(W_est)
    log.info('Solving equality constrained problem ... Done')

    # evaluate
    fdr, tpr, fpr, shd, nnz = utils.count_accuracy(G, G_est)
    log.info('Accuracy: fdr %f, tpr %f, fpr %f, shd %d, nnz %d',
             fdr, tpr, fpr, shd, nnz)
