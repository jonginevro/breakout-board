"""Central configuration: seasons, data URLs, and loaders for the tunable YAML files."""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Seasons -----------------------------------------------------------------
# The season we're projecting breakouts *for*.
UPCOMING_SEASON = 2026
# The most recent completed season we have full stats for (the "prior" baseline).
PRIOR_SEASON = 2025
# How many prior seasons of weekly stats to pull (for multi-year context/efficiency).
HISTORY_SEASONS = [2023, 2024, 2025]

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]

# --- nflverse release asset URLs --------------------------------------------
# Verified live filenames (nflverse renamed the weekly-stats asset to
# `stats_player/stats_player_week_<year>.parquet`).
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"


def stats_url(season: int) -> str:
    return f"{NFLVERSE}/stats_player/stats_player_week_{season}.parquet"


def snaps_url(season: int) -> str:
    return f"{NFLVERSE}/snap_counts/snap_counts_{season}.parquet"


def roster_url(season: int) -> str:
    return f"{NFLVERSE}/rosters/roster_{season}.parquet"


def depth_chart_url(season: int) -> str:
    return f"{NFLVERSE}/depth_charts/depth_charts_{season}.parquet"


DRAFT_PICKS_URL = f"{NFLVERSE}/draft_picks/draft_picks.parquet"
PLAYERS_URL = f"{NFLVERSE}/players/players.parquet"

# --- Sleeper API -------------------------------------------------------------
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_TRENDING_URL = "https://api.sleeper.app/v1/players/nfl/trending/{kind}"


# --- Tunable configs ---------------------------------------------------------
@functools.lru_cache(maxsize=None)
def weights() -> dict:
    with open(CONFIG_DIR / "weights.yaml") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=None)
def manual_catalysts() -> dict:
    path = CONFIG_DIR / "manual_catalysts.yaml"
    if not path.exists():
        return {"teams": {}, "players": {}}
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("teams", {})
    data.setdefault("players", {})
    return data
