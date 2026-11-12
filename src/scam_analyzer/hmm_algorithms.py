"""Discrete-HMM algorithms implemented from scratch (NumPy), with scaling.

Notation (Rabiner, 1989):  N hidden states, M observation symbols, sequence O = o_1 … o_T
    π  (N,)    initial state probabilities         π_i = P(q_1 = i)
    A  (N,N)   transition probabilities            a_ij = P(q_{t+1}=j | q_t=i)
    B  (N,M)   emission probabilities              b_j(k) = P(o_t = k | q_t = j)

Why scaling: the unscaled forward variable α_t(i) is a product of T probabilities and underflows
to 0 for long sequences (the original notebook already reached 1e-226 at T = 121). We normalise
α_t at every step by c_t = Σ_i α_t(i); then  log P(O | λ) = Σ_t log c_t  exactly.
"""
from __future__ import annotations

import numpy as np


def forward(obs, pi, A, B):
    """Scaled forward pass. Returns (alpha_hat (T,N), scales c (T,), log-likelihood)."""
    T, N = len(obs), len(pi)
    alpha = np.zeros((T, N))
    c = np.zeros(T)
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]
    return alpha, c, float(np.log(c).sum())


def backward(obs, A, B, c):
    """Scaled backward pass using the forward scales. Returns beta_hat (T,N)."""
    T, N = len(obs), A.shape[0]
    beta = np.zeros((T, N))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
        beta[t] /= c[t + 1]
    return beta


def posteriors(alpha, beta):
    """γ_t(i) = P(q_t = i | O, λ) — the Forward–Backward state posterior."""
    g = alpha * beta
    return g / g.sum(axis=1, keepdims=True)


def viterbi(obs, pi, A, B):
    """Most likely hidden-state path, in log space. Returns (path (T,), log-prob of that path)."""
    T, N = len(obs), len(pi)
    with np.errstate(divide="ignore"):
        lpi, lA, lB = np.log(pi), np.log(A), np.log(B)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = lpi + lB[:, obs[0]]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + lA          # (from i, to j)
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(N)] + lB[:, obs[t]]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, float(delta[-1].max())


def unscaled_forward_probability(obs, pi, A, B) -> float:
    """The naive (original-notebook) forward probability — kept to demonstrate underflow."""
    alpha = pi * B[:, obs[0]]
    for t in range(1, len(obs)):
        alpha = (alpha @ A) * B[:, obs[t]]
    return float(alpha.sum())
