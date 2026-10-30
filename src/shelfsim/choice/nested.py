"""Nested logit shares and elasticities. sigma = 0 is the plain logit.

With nests g and nesting parameter sigma in [0, 1):

    s_j|g = exp(V_j / (1 - sigma)) / D_g,       D_g = sum_{k in g} exp(V_k / (1 - sigma))
    s_g   = D_g^(1 - sigma) / (1 + sum_h D_h^(1 - sigma))
    s_j   = s_j|g * s_g

sigma near 1 means products in a nest are close substitutes for each other and weak ones for
products outside it, which is what breaks the plain logit's IIA property.
"""

import numpy as np


def nested_shares(v: np.ndarray, nests: np.ndarray, sigma: float) -> np.ndarray:
    scale = 1.0 - sigma
    top = np.max(v / scale)
    expv = np.exp(v / scale - top)
    shares = np.empty_like(v)
    inclusive = {}
    for g in np.unique(nests):
        members = nests == g
        d_g = expv[members].sum()
        shares[members] = expv[members] / d_g
        inclusive[g] = scale * (np.log(d_g) + top)
    log_ig = np.array(list(inclusive.values()))
    m = max(0.0, log_ig.max())
    denominator = np.exp(-m) + np.exp(log_ig - m).sum()
    for g, value in inclusive.items():
        shares[nests == g] *= np.exp(value - m) / denominator
    return shares


def nested_elasticities(
    alpha: float, prices: np.ndarray, shares: np.ndarray, nests: np.ndarray, sigma: float
) -> np.ndarray:
    """E[j, k] = d ln s_j / d ln p_k.

    own:              alpha p_j (1/(1-sigma) - sigma/(1-sigma) s_j|g - s_j)
    same nest, k!=j:  -alpha p_k (sigma/(1-sigma) s_k|g + s_k)
    other nest:       -alpha p_k s_k
    With sigma = 0 these are the plain logit's alpha p_j (1 - s_j) and -alpha p_k s_k.
    """
    within = np.array([shares[i] / shares[nests == nests[i]].sum() for i in range(len(shares))])
    ratio = sigma / (1.0 - sigma)
    same = nests[:, None] == nests[None, :]
    e = np.where(
        same,
        -alpha * prices[None, :] * (ratio * within[None, :] + shares[None, :]),
        -alpha * prices[None, :] * shares[None, :],
    )
    own = alpha * prices * (1.0 / (1.0 - sigma) - ratio * within - shares)
    np.fill_diagonal(e, own)
    return e
