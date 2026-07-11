"""Combine the four factors into a single Breakout Score.

    core   = geomean(opportunity, catalyst, efficiency) ** shape
    value  = value_floor + (1 - value_floor) * suppression
    score  = 100 * core * value * momentum

The geometric mean is the anti-hype guardrail: a player must show up on ALL of
opportunity, catalyst AND efficiency. Miss any one and the product collapses —
which is exactly why a talented-but-buried rookie or a hyped-but-blocked player
doesn't float to the top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import features
from ..data import build as build_mod


def _geomean(a, b, c, eps: float = 1e-3):
    a, b, c = (np.clip(x, eps, 1) for x in (a, b, c))
    return (a * b * c) ** (1 / 3)


def score_players(df: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    if df is None:
        df = build_mod.build(force=force)
    df = features.add_factors(df)
    w = config.weights()

    core = _geomean(df["opportunity"], df["catalyst"], df["efficiency"]) ** w.get("shape", 1.0)
    floor = w.get("value_floor", 0.4)
    value = floor + (1 - floor) * df["suppression"]

    mom_cfg = w.get("momentum", {})
    if mom_cfg.get("enabled", True):
        # momentum_ratio ~1 neutral; >1 rising usage. Fold in Sleeper trending.
        rising = (df["momentum_ratio"] - 1).clip(-0.5, 1.0)
        trend = features._pct(df, "trend_count")  # 0..1 within position
        raw = 0.7 * rising + 0.3 * trend
        df["momentum"] = 1 + mom_cfg.get("max_boost", 0.35) * raw.clip(-1, 1)
    else:
        df["momentum"] = 1.0

    df["breakout_score"] = (100 * core * value * df["momentum"]).round(1)
    df["core"] = (100 * core).round(1)
    df["value_mult"] = value.round(2)
    return df.sort_values("breakout_score", ascending=False).reset_index(drop=True)


# The columns worth surfacing, in display order.
DISPLAY_COLS = [
    "name", "position", "team", "age", "exp", "breakout_score",
    "opportunity", "catalyst", "efficiency", "suppression", "momentum",
    "team_vacated_share", "share_pos", "pos_rank", "draft_pick",
    "ppr_g", "search_rank", "trend_count",
]


if __name__ == "__main__":
    scored = score_players()
    pd.set_option("display.width", 200)
    for pos in ["RB", "WR", "TE", "QB"]:
        top = scored[scored["position"] == pos].head(10)
        print(f"\n===== TOP {pos} BREAKOUTS =====")
        print(top[["name", "team", "age", "exp", "breakout_score", "opportunity",
                   "catalyst", "efficiency", "suppression"]].to_string(index=False))
