"""uv run test_clone.py — the cloning estimator against a closed-form tail.

A walk with Gaussian block scores A_t ~ N(-mu, sigma^2) has
S_L ~ N(-mu L, sigma^2 L), so P[S_L >= aL] = Phi_bar((a + mu) sqrt(L) / sigma)
exactly. The rare case below is p ~ 2e-4: brute force at the same budget sees a
handful of hits at best; the tilt k = (a + mu) / sigma^2 makes it typical.
"""

import math

import numpy as np

from clone import estimate, gktl


def main():
    mu, sigma, L = 0.0, 1.0, 20
    for a, k, N, tol in [(0.1, 0.0, 4000, 0.10), (0.8, 0.8, 4000, 0.35)]:
        p = 0.5 * math.erfc((a + mu) * math.sqrt(L) / sigma / math.sqrt(2))
        ests = []
        for seed in range(8):
            rng = np.random.default_rng(seed)

            def step(states, rng=rng):
                return states, rng.normal(-mu, sigma, len(states))

            res = gktl([None] * N, step, k=k, n_blocks=L, rng=rng)
            ests.append(estimate(res, res["S"] >= a * L, k))
        m, rel = np.mean(ests), np.std(ests) / p
        print(f"a={a} k={k}: exact {p:.3e}  cloning {m:.3e}  (run-to-run rel. spread {rel:.2f})")
        assert abs(m - p) / p < tol, (m, p)
    print("ok")


if __name__ == "__main__":
    main()
