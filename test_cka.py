"""uv run test_cka.py — the gram trick in analyze.cka_matrix is the only
non-obvious maths here, so check it against the explicit definition."""

import torch

from analyze import cka_matrix
from jspace import unembed_gram


def explicit_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


def main():
    torch.manual_seed(0)
    d, vocab = 32, 200
    W = torch.randn(vocab, d)
    J = {0: torch.randn(d, d), 1: torch.randn(d, d), 2: torch.randn(d, d)}
    J[2] = 3.0 * J[1]  # scaling must not change CKA

    class FakeModel:
        _lm_head = torch.nn.Linear(d, vocab, bias=False)

    FakeModel._lm_head.weight.data = W
    G = unembed_gram(FakeModel)

    C = cka_matrix(J, G, "cpu")
    for a in range(3):
        assert abs(C[a][a] - 1.0) < 1e-4, C[a][a]
        for b in range(3):
            assert abs(C[a][b] - C[b][a]) < 1e-5
            want = explicit_cka(W @ J[a], W @ J[b])
            assert abs(C[a][b] - want) < 1e-3, (a, b, C[a][b], want)
    assert abs(C[1][2] - 1.0) < 1e-4, "CKA should be scale-invariant"
    print("ok")


if __name__ == "__main__":
    main()
