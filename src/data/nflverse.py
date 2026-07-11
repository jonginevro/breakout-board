"""Fetch nflverse release data (weekly stats, rosters, depth charts, draft, players)
with a simple on-disk parquet cache so the dashboard is fast and works offline
after the first pull."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .. import config


def _cache_path(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.parquet"


def _load(name: str, url: str, max_age_hours: float, force: bool = False) -> pd.DataFrame:
    """Return a cached parquet if fresh, else download from `url` and cache it."""
    path = _cache_path(name)
    if not force and path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            return pd.read_parquet(path)
    try:
        df = pd.read_parquet(url)
        df.to_parquet(path)
        return df
    except Exception:
        if path.exists():  # network hiccup: fall back to stale cache
            return pd.read_parquet(path)
        raise


# Historical data almost never changes -> long TTL. Rosters/depth move in-season.
def weekly_stats(force: bool = False) -> pd.DataFrame:
    frames = []
    for season in config.HISTORY_SEASONS:
        try:
            frames.append(_load(f"stats_{season}", config.stats_url(season), 24, force))
        except Exception:
            continue  # a season not yet published (e.g. upcoming) is fine
    df = pd.concat(frames, ignore_index=True)
    if "season_type" in df:
        df = df[df["season_type"] == "REG"].copy()
    return df


def players(force: bool = False) -> pd.DataFrame:
    return _load("players", config.PLAYERS_URL, 24 * 7, force)


def draft_picks(force: bool = False) -> pd.DataFrame:
    return _load("draft_picks", config.DRAFT_PICKS_URL, 24 * 7, force)


def roster(season: int | None = None, force: bool = False) -> pd.DataFrame:
    """Latest available seasonal roster (tries upcoming season, falls back)."""
    for yr in ([season] if season else [config.UPCOMING_SEASON, config.PRIOR_SEASON]):
        try:
            df = _load(f"roster_{yr}", config.roster_url(yr), 12, force)
            df["_roster_season"] = yr
            return df
        except Exception:
            continue
    raise RuntimeError("No roster data available from nflverse.")


def depth_charts(season: int | None = None, force: bool = False) -> pd.DataFrame:
    """Latest depth-chart snapshot per player (the file is huge and time-series,
    so we collapse to each player's most recent row)."""
    for yr in ([season] if season else [config.UPCOMING_SEASON, config.PRIOR_SEASON]):
        try:
            df = _load(f"depth_{yr}", config.depth_chart_url(yr), 12, force)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame(columns=["gsis_id", "team", "pos_abb", "pos_rank"])
    if "dt" in df:
        df = df.sort_values("dt").groupby("gsis_id", as_index=False).last()
    keep = [c for c in ["gsis_id", "team", "pos_abb", "pos_grp", "pos_rank", "player_name"] if c in df]
    return df[keep].copy()
