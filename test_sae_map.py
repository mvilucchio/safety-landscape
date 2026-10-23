"""uv run test_sae_map.py — the two claims sae_map.py relies on to avoid ever
building the [F, F] matrix. Small stand-ins for d=768, F=24576."""

import torch

from sae_map import composed_spectrum


def main():
    torch.manual_seed(0)
    d, F = 40, 300
    J = torch.randn(d, d)
    W_enc_t = torch.randn(d, F)      # target encoder  [d, F]
    W_dec_s = torch.randn(F, d)      # source decoder  [F, d]

    M = W_enc_t.T @ J @ W_dec_s.T    # the thing we never build for real  [F, F]
    assert M.shape == (F, F)
    assert torch.linalg.matrix_rank(M).item() == d, "lifting must not add rank"

    S, U, V = composed_spectrum(J, W_enc_t, W_dec_s)
    assert S.shape == (d,) and U.shape == (F, d) and V.shape == (F, d)

    s_true = torch.linalg.svdvals(M)[:d]
    rel = ((S - s_true).abs().max() / s_true[0]).item()
    assert rel < 1e-4, f"spectrum mismatch vs full SVD: {rel}"

    # Singular vectors must reconstruct M, not just its singular values.
    rel_rec = ((U @ torch.diag(S) @ V.T - M).norm() / M.norm()).item()
    assert rel_rec < 1e-4, f"reconstruction off by {rel_rec}"

    # U, V are orthonormal frames in feature space.
    for name, Q in (("U", U), ("V", V)):
        off = (Q.T @ Q - torch.eye(d)).abs().max().item()
        assert off < 1e-4, f"{name} not orthonormal: {off}"

    print("ok")


if __name__ == "__main__":
    main()
