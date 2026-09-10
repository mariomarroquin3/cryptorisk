"""Small numeric helpers shared by the realized-measure models."""

from __future__ import annotations

import numpy as np


def ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs in a 1-D array; leading NaNs get the array median."""
    out = np.asarray(a, float).copy()
    mask = np.isnan(out)
    if not mask.any():
        return out
    idx = np.where(~mask, np.arange(out.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out[mask] = out[idx[mask]]
    if np.isnan(out[0]):
        out[np.isnan(out)] = np.nanmedian(a)
    return out
