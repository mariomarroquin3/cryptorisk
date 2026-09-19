"""LSTM-filtered volatility (Kim & Won, 2018-style), Student-t tail.

An LSTM reads the last ``_SEQ_LEN`` days of (return, squared return, squared
down-return) and outputs log-variance for the next day, trained by Gaussian
quasi-MLE (minimise the Gaussian negative log-likelihood of the realized
return given the predicted variance) -- the same quasi-MLE spirit as every
GARCH-family model here, just with an LSTM's gated recurrence standing in for
the GARCH(1,1) recursion or Realized-SV's CIR/UKF state, so it can pick up
whatever nonlinear dependence on recent returns a fixed parametric recursion
might miss. Once trained, the return distribution is ``loc=0, scale=sqrt(v_hat)``
with a Student-t tail fitted to the in-sample standardized residuals, exactly
like :class:`~cryptorisk.models.har.HAR`.

**Cost.** Unlike every other model in this suite, fitting means a gradient
descent loop, not a closed form or a handful of L-BFGS-B iterations -- too
slow to refit every single day over a multi-year walk-forward. This model is
meant to be run with a coarser ``refit_every`` (see ``config/study.yaml``'s
``walk_forward.refit_every["LSTM-Vol"]``, wired into ``study.run_backtests``); the
walk-forward engine itself reuses the last fitted day's distribution on the
in-between days, exactly as it already does for MS-GARCH.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from cryptorisk.models._dist import student_t_z
from cryptorisk.models.base import Context, EmpiricalDist, ParametricDist, PredictiveDist

_SEQ_LEN = 20
_HIDDEN = 16
_EPOCHS = 80
_LR = 5e-3
_MIN_TRAIN = 120
# Matches config/study.yaml's global `seed` (see random_forest.py's identical
# note) -- pins weight init and any nondeterministic CPU kernel fallback.
_SEED = 20260101


def _features(r: np.ndarray) -> np.ndarray:
    return np.stack([r, r**2, np.minimum(r, 0.0) ** 2], axis=1)  # (n, 3)


def _make_sequences(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n - _SEQ_LEN, _SEQ_LEN, 3)`` feature sequences and their targets
    ``r[_SEQ_LEN:]``; sequence ``i`` covers days ``i .. i+_SEQ_LEN-1`` and
    predicts ``r[i + _SEQ_LEN]`` -- no lookahead, the target day itself never
    enters its own sequence."""
    feats = _features(r)
    n_seq = r.size - _SEQ_LEN
    x = np.empty((n_seq, _SEQ_LEN, 3), dtype=np.float32)
    for i in range(n_seq):
        x[i] = feats[i : i + _SEQ_LEN]
    return x, r[_SEQ_LEN:].astype(np.float32)


def _train_and_forecast(
    x: np.ndarray, y: np.ndarray, x_fcast: np.ndarray, init_log_var: float
) -> tuple[np.ndarray, float]:
    """Train the LSTM by Gaussian quasi-MLE; return the in-sample log-variances
    and the next-day log-variance. torch is imported here, not at module level:
    it is an optional dependency (``pip install -e ".[ml]"``) and the read-only
    API imports the model registry, so a deploy without torch must still start
    (the live refit for this model then fails and the API falls back to the
    frozen backtest row)."""
    import torch
    from torch import nn

    # Single-threaded: CPU intra-op parallelism can reduce floating-point sums
    # in a different order run-to-run, which a fixed seed alone does not pin
    # down -- the suite's determinism test needs bit-for-bit repeatability.
    torch.set_num_threads(1)
    torch.manual_seed(_SEED)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=3, hidden_size=_HIDDEN, batch_first=True)
            self.head = nn.Linear(_HIDDEN, 1)
            # Start at the window's unconditional log-variance: with ~80
            # full-batch steps, a bad initial scale would burn most of them.
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, init_log_var)

        def forward(self, seq: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(seq)
            return self.head(out[:, -1, :]).squeeze(-1)

    x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=_LR)
    for _ in range(_EPOCHS):
        opt.zero_grad()
        log_var = net(x_t)
        # Gaussian quasi-NLL up to the additive constant.
        loss = 0.5 * (log_var + y_t**2 / torch.exp(log_var)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        in_sample = net(x_t).numpy()
        nxt = float(net(torch.from_numpy(x_fcast)).item())
    return in_sample, nxt


class LstmVol:
    """LSTM-filtered conditional variance, Student-t innovation."""

    name = "LSTM-Vol"

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns.astype(np.float32)
        if r.size < _MIN_TRAIN + _SEQ_LEN:
            return EmpiricalDist(ctx.returns)

        x, y = _make_sequences(r)
        x_fcast = _features(r)[-_SEQ_LEN:].reshape(1, _SEQ_LEN, 3).astype(np.float32)
        try:
            log_var_in_sample, log_var_next = _train_and_forecast(
                x, y, x_fcast, float(np.log(np.var(r) + 1e-12))
            )
        except RuntimeError:
            return EmpiricalDist(ctx.returns)

        v_next = float(np.exp(log_var_next))
        if not np.isfinite(v_next) or v_next <= 0 or v_next > 400 * float(np.var(r)):
            return EmpiricalDist(ctx.returns)

        z = y / np.sqrt(np.exp(log_var_in_sample) + 1e-30)
        z = z[np.isfinite(z)]
        nu = float(np.clip(stats.t.fit(z, floc=0)[0], 3.0, 50.0)) if z.size > 50 else 6.0
        ppf, cdf, es = student_t_z(nu)
        return ParametricDist(loc=0.0, scale=float(np.sqrt(v_next)), z_ppf=ppf, z_cdf=cdf, z_es=es)
