"""Turn the raw signals from data/build.py into the four normalized factors:
opportunity, catalyst, efficiency, suppression — each 0..1, ranked WITHIN a
player's position group so we compare like with like. Every sub-signal is kept
on the frame too, so the dashboard can show *why* a player scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def _pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile rank of a column within each position group (0..1)."""
    return df.groupby("position")[col].rank(pct=True).fillna(0.0)


def _wavg(parts: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted average of already-0..1 sub-signals."""
    total = sum(weights.get(k, 0) for k in parts) or 1.0
    out = sum(parts[k] * weights.get(k, 0) for k in parts) / total
    return out.clip(0, 1)


def _age_window_score(df: pd.DataFrame) -> pd.Series:
    windows = config.weights()["age_windows"]

    def score(row):
        start, peak, end = windows.get(row["position"], [22, 25, 28])
        a = row["age"]
        if pd.isna(a):
            return 0.4
        if a <= peak:
            return float(np.clip((a - (start - 2)) / max(peak - (start - 2), 1), 0, 1))
        return float(np.clip(1 - (a - peak) / max((end + 2) - peak, 1), 0, 1))

    return df.apply(score, axis=1)


_RANK_DESERVED = {1: 0.70, 2: 0.35, 3: 0.15}
_RANK_PROXIMITY = {1: 1.0, 2: 0.6, 3: 0.35}


def _manual_boost(df: pd.DataFrame) -> pd.Series:
    cat = config.manual_catalysts()
    teams, players = cat.get("teams") or {}, cat.get("players") or {}
    lower = {k.lower(): v for k, v in players.items()}

    def score(row):
        total = 0.0
        for _, v in (teams.get(row["team"], {}) or {}).items():
            total += float(v)
        for _, v in (lower.get(str(row["name"]).lower(), {}) or {}).items():
            total += float(v)
        return min(total, 1.0)

    return df.apply(score, axis=1)


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = config.weights()

    # ---- OPPORTUNITY ---------------------------------------------------------
    df["_headroom"] = (1 - df["share_pos"]).clip(0, 1)
    deserved = df["pos_rank"].map(_RANK_DESERVED).fillna(0.05)
    df["_depth_gap"] = (deserved - df["share_pos"]).clip(lower=0)
    opp = {
        "team_vacated_share": _pct(df, "team_vacated_share"),
        "personal_headroom": _pct(df, "_headroom"),
        "depth_gap": _pct(df, "_depth_gap"),
    }
    df["opportunity"] = _wavg(opp, w["opportunity"])

    # ---- CATALYST ------------------------------------------------------------
    df["_draft_cap"] = (1 - (df["draft_pick"] - 1) / 261).clip(0, 1)
    df["_age_win"] = _age_window_score(df)
    df["_second_year"] = df["exp"].map({1: 1.0, 2: 0.55, 0: 0.35}).fillna(0.15)
    df["_proximity"] = df["pos_rank"].map(_RANK_PROXIMITY).fillna(0.20)
    df["_manual"] = _manual_boost(df)
    cat = {
        "draft_capital": df["_draft_cap"],
        "age_window": df["_age_win"],
        "second_year_leap": df["_second_year"],
        "depth_proximity": df["_proximity"],
        "manual_boost": df["_manual"],
    }
    df["catalyst"] = _wavg(cat, w["catalyst"])

    # ---- EFFICIENCY ----------------------------------------------------------
    # advanced signal is position-specific (usage quality)
    adv = pd.Series(np.nan, index=df.index)
    is_rec = df["position"].isin(["WR", "TE"])
    adv[is_rec] = df.loc[is_rec, "wopr"]
    adv[df["position"] == "RB"] = df.loc[df["position"] == "RB", "ypt"]
    adv[df["position"] == "QB"] = df.loc[df["position"] == "QB", "ppr_g"]
    df["_adv_raw"] = adv
    eff = {
        "per_touch": _pct(df, "ppr_g"),
        "advanced": _pct(df, "_adv_raw"),
        "draft_capital": df["_draft_cap"],
    }
    df["efficiency"] = _wavg(eff, w["efficiency"])

    # ---- SUPPRESSION -- log-gaussian around the value "sweet spot" ------------
    # Peaks for cheap-but-rostered players; ~0 for elite (priced in) AND for
    # off-radar/sentinel players (no market to spike).
    vcfg = w.get("value", {"peak_rank": 175, "sigma": 0.55})
    logr = np.log10(df["search_rank"].clip(lower=1))
    peak = np.log10(max(vcfg["peak_rank"], 1))
    df["suppression"] = np.exp(-((logr - peak) ** 2) / (2 * vcfg["sigma"] ** 2))

    return df
