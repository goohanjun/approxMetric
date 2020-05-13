import numpy as np
from pyemd import emd, emd_with_flow
from sklearn.metrics import pairwise_distances
import time


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k - 1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def compute_rwmd(hist_1, hist_2, dists):
    idx_1, idx_2 = hist_1.nonzero()[0], hist_2.nonzero()[0]
    p_1, p_2 = hist_1[idx_1], hist_2[idx_2]
    dists_sel = dists[np.ix_(idx_1, idx_2)]
    # print("p_1", p_1)
    # print("p_2", p_2)
    # print("dists_sel\n", dists_sel)
    # print("dists_sel min")
    # dists_sel[dists_sel == 0.] = 1e10
    # print(dists_sel)
    # print(np.min(dists_sel[np.nonzero(dists_sel)], ))
    # print("case 1", p_1 * np.min(dists_sel, axis=1))
    # print("case 2", p_2 * np.min(dists_sel, axis=0))

    cost_1 = np.sum(p_1 * np.min(dists_sel, axis=1))
    cost_2 = np.sum(p_2 * np.min(dists_sel, axis=0))
    return max(cost_1, cost_2)


def compute_omr(hist_1, hist_2, dists):
    idx_1, idx_2 = hist_1.nonzero()[0], hist_2.nonzero()[0]
    p, q = hist_1[idx_1], hist_2[idx_2]
    dists_sel = dists[np.ix_(idx_1, idx_2)]

    def asym_omr(p_1, p_2, dists):
        cost = 0.
        for i, p_i in enumerate(p_1):
            p_tmp = p_i
            top_k_idx = np.argpartition(dists[i, :], 2)[:2]
            i_dist = dists[i, top_k_idx]
            s = [x for _, x in sorted(zip(i_dist, top_k_idx))]

            if dists[i, s[0]] == 0.:
                r = min(p_tmp, p_2[s[0]])
                p_tmp -= r
                cost += p_tmp * dists[i, s[1]]
            else:
                cost += p_tmp * dists[i, s[0]]
        return cost
    return max(asym_omr(p, q, dists_sel), asym_omr(q, p, dists_sel.T))

def compute_ict(hist_1, hist_2, dists):
    """
    Compute lower bound of EMD using ICT
    http://proceedings.mlr.press/v97/atasu19a/atasu19a.pdf
    time complexity O(n^2) algorithm

    :param hist_1: [# of word]
    :param hist_2: [# of word]
    :param dists: [# of word, # of word]
    :return: ICT value
    """
    idx_1, idx_2 = hist_1.nonzero()[0], hist_2.nonzero()[0]
    p, q = hist_1[idx_1], hist_2[idx_2]
    dists_sel = dists[np.ix_(idx_1, idx_2)]

    def asym_ict(p_1, p_2, dists):
        cost = 0.
        for i, p_i in enumerate(p_1):
            p_tmp = p_i
            s = np.argsort(dists[i, :])  # smaller comes first
            # print("dist_sel@", i, dists_sel[i, :])
            # print("s@", i, s)
            for j in s:
                r = min(p_tmp, p_2[j])
                p_tmp -= r
                cost += r * dists[i, j]
                if p_tmp <= 0.: break
        return cost
    return max(asym_ict(p, q, dists_sel), asym_ict(q, p, dists_sel.T))

def compute_act(hist_1, hist_2, dists_matrix, k=3):
    """
    Compute lower bound of EMD using Approximate ICT
    http://proceedings.mlr.press/v97/atasu19a/atasu19a.pdf
    time complexity O(n k) algorithm
    :param hist_1: [# of word]
    :param hist_2: [# of word]
    :param dists: [# of word, # of word]
    :return: ICT value
    """
    idx_1, idx_2 = hist_1.nonzero()[0], hist_2.nonzero()[0]
    p, q = hist_1[idx_1], hist_2[idx_2]
    dists_sel = dists_matrix[np.ix_(idx_1, idx_2)]

    def asym_act(p_1, p_2, dists):
        cost = 0.
        for i, p_i in enumerate(p_1):
            p_tmp = p_i

            if len(idx_2) > k:
                top_k_idx = np.argpartition(dists[i, :], k)[:k]
                i_dist = dists[i, top_k_idx]
                top_k_idx = [x for _, x in sorted(zip(i_dist, top_k_idx))]
            else:
                i_dist = dists[i, :]
                top_k_idx = np.argsort(dists[i, :])
            # print("i_dist", i_dist)
            # print("top_k_idx", top_k_idx)
            for l_idx in top_k_idx:
                r = min(p_tmp, p_2[l_idx])
                p_tmp -= r
                cost += r * dists[i, l_idx]
            if p_tmp > 0: cost += p_tmp * dists[i, top_k_idx[-1]]
        return cost
    return max(asym_act(p, q, dists_sel), asym_act(q, p, dists_sel.T))


def compute_greenkhorn_0_1(a, b, M):
    return compute_greenkhorn(a, b, M, timeThr=0.1)


def compute_greenkhorn_0_01(a, b, M):
    return compute_greenkhorn(a, b, M, timeThr=0.01)


def compute_greenkhorn(a, b, M, timeThr, reg=1., numItermax=10000, stopThr=1e-9, verbose=False, log=False):
    r"""
    https://github.com/PythonOT/POT/blob/master/ot/bregman.py
    Solve the entropic regularization optimal transport problem and return the OT matrix
    The algorithm used is based on the paper
    Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
        by Jason Altschuler, Jonathan Weed, Philippe Rigollet
        appeared at NIPS 2017
    which is a stochastic version of the Sinkhorn-Knopp algorithm [2].
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)
    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,) or ndarray, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    Examples
    --------
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
       [22] J. Altschuler, J.Weed, P. Rigollet : Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration, Advances in Neural Information Processing Systems (NIPS) 31, 2017
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
    """

    start_time = time.time()

    # Pick
    idx_1, idx_2 = a.nonzero()[0], b.nonzero()[0]
    a, b = a[idx_1], b[idx_2]
    M = M[np.ix_(idx_1, idx_2)]

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dists = M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty_like(M)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    u = np.full(dim_a, 1. / dim_a)
    v = np.full(dim_b, 1. / dim_b)
    G = u[:, np.newaxis] * K * v[np.newaxis, :]

    viol = G.sum(1) - a
    viol_2 = G.sum(0) - b
    stopThr_val = 1
    while True:
        i_1 = np.argmax(np.abs(viol))
        i_2 = np.argmax(np.abs(viol_2))
        m_viol_1 = np.abs(viol[i_1])
        m_viol_2 = np.abs(viol_2[i_2])
        stopThr_val = np.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            u[i_1] = a[i_1] / (K[i_1, :].dot(v) + 1e-6)
            G[i_1, :] = u[i_1] * K[i_1, :] * v

            viol[i_1] = u[i_1] * K[i_1, :].dot(v) - a[i_1]
            viol_2 += (K[i_1, :].T * (u[i_1] - old_u) * v)

        else:
            old_v = v[i_2]
            v[i_2] = b[i_2] / (K[:, i_2].T.dot(u) + 1e-6)
            G[:, i_2] = u * K[:, i_2] * v[i_2]
            # aviol = (G@one_m - a)
            # aviol_2 = (G.T@one_n - b)
            viol += (-old_v + v[i_2]) * K[:, i_2] * u
            viol_2[i_2] = v[i_2] * K[:, i_2].dot(u) - b[i_2]

        if (time.time() - start_time) > timeThr: break
        if stopThr_val <= stopThr: break

    # G is a flow matrix
    dist = np.sum(G * dists)
    if np.isnan(dist): raise Exception("NaN occured in greenkhorn()")
    return dist


def compute_UB_G(hist_1, hist_2, dists_matrix):
    """
    :param hist_1 (q):
    :param hist_2 (p):
    :param dists:
    :return:
    """
    idx_1, idx_2 = hist_1.nonzero()[0], hist_2.nonzero()[0]
    dists_sel = dists_matrix[np.ix_(idx_1, idx_2)]

    # current_cost
    current_cost = 0.

    # max_flow calculation
    q_cap, p_cap = hist_1[idx_1], hist_2[idx_2]
    q_flow, p_flow = np.zeros_like(q_cap), np.zeros_like(p_cap)

    max_flow_cost = 0.
    flow_cnt, p_flag = 0, True
    while True:
        feasible_q = np.nonzero((q_cap - q_flow) > 0)[0]
        if flow_cnt == 0 and p_flag == False: break
        if len(feasible_q) == 0: break
        q_i = feasible_q[0]

        flow_cnt, p_flag = 0, True
        while q_cap[q_i] > q_flow[q_i]:
            feasible_p = ((p_cap - p_flow) > 0).nonzero()[0]
            if len(feasible_p) == 0: p_flag = False; break
            feasible_p_idx = np.argmin(dists_sel[q_i, feasible_p])
            p_j = feasible_p[feasible_p_idx]

            flow = min(q_cap[q_i] - q_flow[q_i], p_cap[p_j] - p_flow[p_j])
            q_flow[q_i] += flow
            p_flow[p_j] += flow
            max_flow_cost += flow * dists_sel[q_i, p_j]
            flow_cnt += 1
    return current_cost + max_flow_cost


def compute_hmean_rwmd_UB_G(hist_1, hist_2, dists):
    l = compute_rwmd(hist_1, hist_2, dists)
    u = compute_UB_G(hist_1, hist_2, dists)
    return 2 * l * u / (l + u)


def compute_hmean_ict_UB_G(hist_1, hist_2, dists):
    l = compute_ict(hist_1, hist_2, dists)
    u = compute_UB_G(hist_1, hist_2, dists)
    return 2 * l * u / (l + u)


if __name__ == "__main__":
    print("Testing...")
    hist_1, hist_2 = np.asarray([0.2, 0.3, 0.5]), np.asarray([0.5, 0.1, 0.4])
    dists = np.asarray([[0, 1., 4.],
                        [1., 0., 3.],
                        [4., 3., 0.]])

    dist, flow = emd_with_flow(hist_1, hist_2, dists)
    print(" EMD   distance: ", dist)
    # for ii, jj in k_largest_index_argsort(np.asarray(flow), 6):
    #     print(f"{ii} -> {jj}, flow: {flow[ii][jj]:.2f}")

    funcs = [compute_greenkhorn,  # NIPS'17
             compute_rwmd, compute_omr, compute_act, compute_ict,  # ICML'19
             compute_UB_G,  # VLDB'13
             compute_hmean_rwmd_UB_G, compute_hmean_ict_UB_G]  # TKDE'19

    print("\nApprox distances")
    for dist_func in funcs:
        print(f"{dist_func.__name__.replace('compute_', ''):<15} distance: {dist_func(hist_1, hist_2, dists):.2f}")
