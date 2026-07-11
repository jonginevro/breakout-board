"""Sleeper API: current market value proxy (search_rank) and the live 'trending
adds' signal that powers the in-season momentum factor. Free, no auth."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .. import config


def _cache(name: str) -> Path:
    return config.CACHE_DIR / name


def players(max_age_hours: float = 12, force: bool = False) -> pd.DataFrame:
    """One row per NFL player Sleeper knows about, keyed for joining to nflverse
    via `gsis_id`. `search_rank` is Sleeper's popularity/value ordering
    (lower = more valuable); we invert it into a 'cheapness' signal later."""
    path = _cache("sleeper_players.json")
    fresh = path.exists() and (time.time() - path.stat().st_mtime) / 3600 < max_age_hours
    if force or not fresh:
        try:
            raw = requests.get(config.SLEEPER_PLAYERS_URL, timeout=60).json()
            path.write_text(json.dumps(raw))
        except Exception:
            if not path.exists():
                raise
            raw = json.loads(path.read_text())
    else:
        raw = json.loads(path.read_text())

    rows = []
    for pid, p in raw.items():
        if not isinstance(p, dict):
            continue
        rows.append({
            "sleeper_id": pid,
            "gsis_id": p.get("gsis_id"),
            "sleeper_name": p.get("full_name"),
            "sleeper_team": p.get("team"),
            "sleeper_pos": p.get("position"),
            "search_rank": p.get("search_rank"),
            "sleeper_status": p.get("status"),
        })
    df = pd.DataFrame(rows)
    df["search_rank"] = pd.to_numeric(df["search_rank"], errors="coerce")
    return df


def trending(kind: str = "add", lookback_hours: int = 48, limit: int = 300,
             max_age_hours: float = 3, force: bool = False) -> pd.DataFrame:
    """Live waiver signal: players being added across Sleeper leagues right now.
    Returns sleeper_id + add count. Cached briefly so repeated dashboard loads
    don't hammer the API."""
    path = _cache(f"sleeper_trending_{kind}.json")
    fresh = path.exists() and (time.time() - path.stat().st_mtime) / 3600 < max_age_hours
    if force or not fresh:
        try:
            url = config.SLEEPER_TRENDING_URL.format(kind=kind)
            raw = requests.get(url, params={"lookback_hours": lookback_hours, "limit": limit},
                               timeout=30).json()
            path.write_text(json.dumps(raw))
        except Exception:
            raw = json.loads(path.read_text()) if path.exists() else []
    else:
        raw = json.loads(path.read_text())
    if not raw:
        return pd.DataFrame(columns=["sleeper_id", "trend_count"])
    return pd.DataFrame([{"sleeper_id": r["player_id"], "trend_count": r.get("count", 0)}
                         for r in raw])
