"""One-line idea, family, and known limitations per registered model.

Shared source of truth for the human-readable side of each model: used to
build the ``docs/model_cards/*.md`` (see ``study/report.py``) and the API's
``/models/info`` (see ``api/app.py``), so the same text a supervisor reads in
the write-up is what a web user sees hovering a model name. Lives in
``models/`` (not ``study/``) specifically so the lightweight API layer can
import it without pulling in ``study.report``'s matplotlib dependency.
"""

from __future__ import annotations

# one-line idea + model-specific limitations, keyed by registry name.
MODEL_NOTES: dict[str, tuple[str, list[str]]] = {
    "HS": (
        "Empirical quantile of the last w returns; no distributional assumption.",
        [
            "Reacts slowly to a regime change (the whole window must roll over).",
            "The tail is only as resolved as `alpha*w` observations.",
        ],
    ),
    "AWHS": (
        "Historical Simulation with exponentially decaying weights (half-life 125d).",
        [
            "Effective sample size ~ H/ln2 << w, so the tail estimate is noisier.",
            "Adapts to the recency of the tail, not its shape.",
        ],
    ),
    "EWMA": (
        "RiskMetrics: one-parameter recursive variance, Gaussian (or fixed-nu t) tail.",
        [
            "Integrated (alpha+beta=1): no mean reversion, no finite unconditional variance.",
            "The Gaussian tail is too thin at alpha=0.01 -- see the ES test.",
        ],
    ),
    "GARCH-t": (
        "GARCH(1,1) variance recursion, standardized Student-t innovations.",
        [
            "Symmetric response to good and bad news.",
            "One regime: a single (alpha, beta) for calm and crisis alike.",
        ],
    ),
    "GJR-GARCH-t": (
        "GARCH-t plus a leverage term (bad news raises variance more).",
        ["The asymmetry is a single extra parameter, constant over time."],
    ),
    "EGARCH-t": (
        "Models log-variance, so positivity is free and the sign asymmetry is direct.",
        [
            "The log-variance recursion can collapse toward 0 (guarded by a vol floor) "
            "or blow up near the non-stationary boundary (vol ceiling).",
            "Absorbs excess kurtosis into the recursion, leaving nu large and the "
            "1% quantile thin.",
        ],
    ),
    "FHS": (
        "GARCH filter, then the empirical tail of the standardized residuals.",
        [
            "Inherits any misspecification of the GARCH volatility dynamics.",
            "The residual tail is still a finite sample -- sparse at alpha=0.01.",
        ],
    ),
    "GARCH-EVT": (
        "GARCH filter, then a Generalized Pareto fit to the residual tail (McNeil-Frey).",
        [
            "Sensitive to the threshold (fixed here at the 90th loss percentile).",
            "The GPD shape parameter is estimated from few exceedances.",
        ],
    ),
    "Jump-Diffusion": (
        "Brownian part plus a compound-Poisson jump component (Merton).",
        [
            "Jump intensity and size are constant -- no clustering of jumps.",
            "VaR/ES come from a fixed-seed Monte-Carlo sample, so a small "
            "quantisation error remains.",
        ],
    ),
    "HAR-RV": (
        "OLS of realized variance on its daily / weekly / monthly averages (Corsi).",
        [
            "Targets the conditional *mean* of RV; RV is right-skewed, so the model "
            "runs thin for a lower return quantile.",
            "Linear -- no volatility-of-volatility term.",
        ],
    ),
    "HARQ": (
        "HAR-RV with the daily lag shrunk on days when RV was measured noisily.",
        [
            "Best variance forecaster, near-worst for VaR/ES: the tail is a point "
            "forecast with a thin t on top.",
            "The RQ interaction is z-scored using window-wide moments (mild leak).",
        ],
    ),
    "Realized-GARCH": (
        "GARCH driven by observed RV, with a measurement equation tying "
        "the two (Hansen-Huang-Shek).",
        [
            "Eight parameters -- more to estimate than plain GARCH.",
            "Assumes a Gaussian measurement error for log RV.",
        ],
    ),
    "Realized-SV": (
        "Heston-style CIR variance filtered with an Unscented Kalman Filter "
        "from the realized measure alone (log RV_t = zeta + phi*log V_t + "
        "u_t, phi fitted not fixed at 1), plus a GJR-style lagged leverage "
        "term. The return does not update the filtered state -- it scores "
        "its own exact Student-t log-density given the predicted V instead "
        "of going through the approximate Harvey-Ruiz-Shephard log-square "
        "trick tried during development, which stayed biased even after "
        "correcting its Gaussian constants for fat tails. VaR/ES come from "
        "a GPD tail (McNeil-Frey, 2000, same construction as GARCH-EVT) "
        "fitted to the filtered window's own standardized residuals, not "
        "the fitted Student-t directly -- a single nu from the whole-sample "
        "likelihood undersold the most extreme moves.",
        [
            "Leverage is lagged (gamma * I(r_{t-1}<0) * resid_{t-1}^2 in the "
            "CIR drift), not Heston's same-day correlated shocks (rho != 0) "
            "-- avoids a same-day causality problem in the filter, at the "
            "cost of a one-day-slower asymmetric response.",
            "Quasi-MLE via an approximate filter likelihood (Gaussian on "
            "the RV channel only), not the exact CIR transition density.",
        ],
    ),
    "GARCH-X": (
        "GARCH-t with lagged RV added to the variance equation.",
        [
            "Degrades to plain GARCH-t (gamma=0) when the realized block is absent.",
            "The exogenous term is a single contemporaneous lag.",
        ],
    ),
    "CAViaR-SAV": (
        "Autoregression of the quantile itself (symmetric absolute value), fit by the tick loss.",
        [
            "No full density: no PIT / Berkowitz, and ES is a scaled quantile.",
            "Symmetric news impact -- unlike AS it cannot react differently to good vs bad days.",
            "Fit separately per alpha; cross-alpha monotonicity is repaired ex-post.",
        ],
    ),
    "CAViaR-AS": (
        "Autoregression of the quantile itself (asymmetric slope), fit by the tick loss.",
        [
            "No full density: no PIT / Berkowitz, and ES is a scaled quantile.",
            "Fit separately per alpha; cross-alpha monotonicity is repaired ex-post.",
        ],
    ),
    "CAViaR-X-AS": (
        "CAViaR-AS plus sqrt(RV_{t-1}) as an exogenous quantile driver.",
        [
            "Same density / ES caveats as CAViaR-AS.",
            "Adds one parameter estimated by a derivative-free search.",
        ],
    ),
    "MS-GARCH": (
        "Two-regime Markov-switching GARCH mixture, walk-forward in R.",
        [
            "The walk-forward regime probability has ~no out-of-sample signal "
            "(see the regime-identification section); the layer is descriptive.",
            "Served from a cache; dates without a prediction fall back to empirical.",
        ],
    ),
    "RF-QR": (
        "Quantile Regression Forest (Meinshausen, 2006): a random forest whose "
        "leaves keep their full set of training returns instead of collapsing "
        "to a mean, so VaR/ES are weighted empirical quantiles of that pooled "
        "distribution -- no parametric tail assumption, unlike every "
        "GARCH-family model here. Features are HAR-style rolling means of "
        "squared returns (plus log RV when available), not a fitted variance "
        "recursion.",
        [
            "No explicit time-series dynamics -- the forest sees lagged "
            "features, not a state that evolves (no persistence parameter "
            "like GARCH's alpha+beta).",
            "The tail is only as resolved as the leaves that survive to the "
            "test point; extreme quantiles rely on however few training "
            "returns land there.",
            "VaR and ES are built from returns observed in the window, so "
            "they can never be more extreme than the window's worst day: a "
            "new record loss is always a violation.",
        ],
    ),
    "LSTM-Vol": (
        "An LSTM reads the last 20 days of (return, squared return, squared "
        "down-return) and outputs next-day log-variance, trained by Gaussian "
        "quasi-MLE -- the same quasi-MLE principle as the GARCH family, with "
        "a gated recurrent net standing in for the fixed GARCH(1,1) "
        "recursion. VaR/ES come from a Student-t tail fitted to the in-sample "
        "standardized residuals, same as HAR-RV.",
        [
            "The weights are retrained only every 20 days (gradient descent, "
            "not a closed form or a handful of L-BFGS-B steps); between "
            "retrains the stored network still runs daily on the newest 20 "
            "returns, but its parameters are up to 19 days stale -- coarser "
            "than every other model here, which refit daily.",
            "The inputs are raw returns (squared returns are ~1e-4), so "
            "permutation importance puts ~94% on the signed return: the "
            "network barely uses its squared-return channels. Standardizing "
            "them was tried and made calibration worse (hit rates ~4-6% at a "
            "2.5% target), so this is an open modelling limitation.",
            "A single fixed architecture/seed per window: no hyperparameter "
            "search, no ensembling across initializations.",
        ],
    ),
}

METHOD_FAMILY = {
    "HS": "Non-parametric",
    "AWHS": "Non-parametric",
    "EWMA": "Exponential smoothing",
    "GARCH-t": "Single-regime GARCH",
    "GJR-GARCH-t": "Single-regime GARCH",
    "EGARCH-t": "Single-regime GARCH",
    "FHS": "Semiparametric tail",
    "GARCH-EVT": "Semiparametric tail",
    "Jump-Diffusion": "Discontinuous",
    "HAR-RV": "Realized-measure",
    "HARQ": "Realized-measure",
    "Realized-GARCH": "Realized-measure",
    "Realized-SV": "Stochastic volatility",
    "GARCH-X": "Exogenous / conditional",
    "CAViaR-SAV": "Exogenous / conditional",
    "CAViaR-AS": "Exogenous / conditional",
    "CAViaR-X-AS": "Exogenous / conditional",
    "MS-GARCH": "Regime-switching",
    "RF-QR": "Machine learning",
    "LSTM-Vol": "Machine learning",
}

# The 4-asset portfolio study (study/run_portfolio.py) scores copula
# *dependence* choices, not registry models: marginals are always
# GARCH(1,1)-t + FHS residual inversion, only the joint tail structure
# varies. Keyed by the exact "Copula-<family>" name run_portfolio.py emits.
COPULA_NOTES: dict[str, tuple[str, list[str]]] = {
    "Copula-independence": (
        "Treats each asset's tail risk as independent -- ignores that "
        "crypto assets crash together.",
        [
            "Understates basket tail risk whenever assets are actually "
            "correlated in the tail, which they are.",
            "Included as the baseline that shows why a copula is needed "
            "at all, not as a candidate worth using.",
        ],
    ),
    "Copula-gaussian": (
        "A Gaussian copula: correlated, but with no extra tail dependence "
        "beyond that correlation.",
        ["Underestimates joint crashes -- the Gaussian copula's tail "
         "dependence is exactly zero, however high the correlation."],
    ),
    "Copula-student_t": (
        "A Student-t copula (fixed degrees of freedom): correlated *and* "
        "prone to joint extreme moves, unlike the Gaussian copula.",
        ["The degrees of freedom are fixed, not fitted per window."],
    ),
    "Copula-clayton": (
        "A Clayton copula: asymmetric tail dependence, strongest in the "
        "joint-crash (lower) tail specifically.",
        [
            "Falls back to independence when the fitted theta is <= 0 "
            "(no positive lower-tail dependence detected in that window).",
        ],
    ),
}
