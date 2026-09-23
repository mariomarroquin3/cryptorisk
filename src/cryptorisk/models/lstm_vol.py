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

**Cost and the refit schedule.** Fitting is a gradient-descent loop, not a
closed form or a handful of L-BFGS-B iterations, so refitting the *weights*
every day over a multi-year walk-forward is too slow. The weights are
retrained every ``retrain_days`` (default 20) and cached; between retrains the
stored network is still *run* every day on the newest ``_SEQ_LEN`` returns, so
the variance forecast reacts to new data daily, like every other model here.
(The engine's ``refit_every`` must stay 1 for this model: using it instead
freezes the whole forecast for the refit period, which is what an earlier
version did.)
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


def _train_net(x: np.ndarray, y: np.ndarray, init_log_var: float, *, importance_repeats: int = 0):
    """Train the LSTM by Gaussian quasi-MLE. Returns ``(net, in-sample
    log-variances, permutation importance or None)``. The importance is, for
    every (lag, feature) input cell, the rise in the Gaussian quasi-NLL over the
    training sequences when that one column is shuffled across sequences
    (averaged over the repeats); shape ``(_SEQ_LEN, 3)``, row 0 = oldest lag.
    torch is imported here, not at module level: it is an optional dependency
    (``pip install -e ".[ml]"``) and the read-only API imports the model
    registry, so a deploy without torch must still start (the live refit for
    this model then fails and the API falls back to the frozen backtest row)."""
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
    net.eval()
    with torch.no_grad():
        in_sample = net(x_t).numpy()
        imp = None
        if importance_repeats > 0:

            def nll(seq: torch.Tensor) -> float:
                lv = net(seq)
                return float((0.5 * (lv + y_t**2 / torch.exp(lv))).mean())

            base = nll(x_t)
            gen = torch.Generator().manual_seed(_SEED)
            imp = np.zeros((_SEQ_LEN, 3))
            for lag in range(_SEQ_LEN):
                for f in range(3):
                    for _ in range(importance_repeats):
                        xp = x_t.clone()
                        xp[:, lag, f] = x_t[torch.randperm(x_t.shape[0], generator=gen), lag, f]
                        imp[lag, f] += (nll(xp) - base) / importance_repeats
    return net, in_sample, imp


def _predict_log_var(net, x_fcast: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        return float(net(torch.from_numpy(x_fcast)).item())


class LstmVol:
    """LSTM-filtered conditional variance, Student-t innovation. Weights are
    retrained every ``retrain_days`` calendar days per asset and cached on the
    instance; the forecast itself is recomputed every day."""

    name = "LSTM-Vol"

    def __init__(self, retrain_days: int = 20) -> None:
        self.retrain_days = retrain_days
        self._cache: dict[str, dict] = {}

    def _usable_cache(self, ctx: Context) -> dict | None:
        """The cached fit, if it is still fresh and provably belongs to this
        very return series: it must have been trained on a window ending at a
        date that is in ``ctx`` with the same return, no later than ``asof``,
        and less than ``retrain_days`` old."""
        c = self._cache.get(ctx.asset)
        if c is None:
            return None
        asof = np.datetime64(ctx.asof, "D")
        age = int((asof - c["asof"]) / np.timedelta64(1, "D"))
        if age < 0 or age >= self.retrain_days:
            return None
        dates = np.asarray(ctx.dates).astype("datetime64[D]")
        i = int(np.searchsorted(dates, c["asof"]))
        if i >= dates.size or dates[i] != c["asof"] or ctx.returns[i] != c["last_return"]:
            return None
        return c

    def fit_predict(self, ctx: Context) -> PredictiveDist:
        r = ctx.returns.astype(np.float32)
        if r.size < _MIN_TRAIN + _SEQ_LEN:
            return EmpiricalDist(ctx.returns)
        x_fcast = _features(r)[-_SEQ_LEN:].reshape(1, _SEQ_LEN, 3).astype(np.float32)

        try:
            c = self._usable_cache(ctx)
            if c is None:
                x, y = _make_sequences(r)
                net, in_sample, _ = _train_net(x, y, float(np.log(np.var(r) + 1e-12)))
                z = y / np.sqrt(np.exp(in_sample) + 1e-30)
                z = z[np.isfinite(z)]
                nu = float(np.clip(stats.t.fit(z, floc=0)[0], 3.0, 50.0)) if z.size > 50 else 6.0
                c = {
                    "net": net,
                    "nu": nu,
                    "asof": np.datetime64(ctx.asof, "D"),
                    "last_return": ctx.returns[-1],
                }
                self._cache[ctx.asset] = c
            log_var_next = _predict_log_var(c["net"], x_fcast)
        except RuntimeError:
            return EmpiricalDist(ctx.returns)

        v_next = float(np.exp(log_var_next))
        if not np.isfinite(v_next) or v_next <= 0 or v_next > 400 * float(np.var(r)):
            return EmpiricalDist(ctx.returns)
        ppf, cdf, es = student_t_z(c["nu"])
        return ParametricDist(loc=0.0, scale=float(np.sqrt(v_next)), z_ppf=ppf, z_cdf=cdf, z_es=es)

    def explain(self, ctx: Context, repeats: int = 3) -> dict | None:
        """Permutation importance of each (lag, feature) input cell for a
        network freshly fitted on this window; ``importance[lag - 1][feature]``
        uses lag 1 = yesterday. ``None`` when the window is too short."""
        r = ctx.returns.astype(np.float32)
        if r.size < _MIN_TRAIN + _SEQ_LEN:
            return None
        x, y = _make_sequences(r)
        _, _, imp = _train_net(x, y, float(np.log(np.var(r) + 1e-12)), importance_repeats=repeats)
        return {
            "features": ["return", "squared return", "squared down-return"],
            "importance": imp[::-1].tolist(),
        }
