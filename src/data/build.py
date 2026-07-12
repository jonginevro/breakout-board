"""Assemble a single player-level table carrying the RAW signals the model
needs. Normalization + weighting happens later in model/features.py — this file
only gathers and joins facts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import nflverse, sleeper

SEASON_START = pd.Timestamp(f"{config.UPCOMING_SEASON}-09-01")


def _opp_unit(row) -> float:
    """Opportunity 'workload unit' by position — the thing a breakout captures
    more of. RB: carries+catches, WR/TE: targets, QB: pass attempts + carries."""
    pos = row["position"]
    c = row.get("carries", 0) or 0
    if pos == "RB":
        return c + (row.get("receptions", 0) or 0)
    if pos in ("WR", "TE"):
        return row.get("targets", 0) or 0
    if pos == "QB":
        return (row.get("attempts", 0) or 0) + c
    return c + (row.get("receptions", 0) or 0)


def _prior_aggregate(stats: pd.DataFrame, season: int) -> pd.DataFrame:
    df = stats[stats["season"] == season].copy()
    if df.empty:
        return pd.DataFrame()
    # Each player's primary team that season = the one he played the most weeks for.
    team_by_weeks = (df.groupby(["player_id", "team"]).size().reset_index(name="n")
                       .sort_values("n").groupby("player_id").last()["team"])
    agg = df.groupby("player_id").agg(
        name=("player_display_name", "last"),
        position=("position", "last"),
        games=("week", "nunique"),
        carries=("carries", "sum"),
        receptions=("receptions", "sum"),
        targets=("targets", "sum"),
        attempts=("attempts", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        passing_yards=("passing_yards", "sum"),
        ppr=("fantasy_points_ppr", "sum"),
        target_share=("target_share", "mean"),
        wopr=("wopr", "mean"),
        rush_epa=("rushing_epa", "mean"),
        rec_epa=("receiving_epa", "mean"),
    ).reset_index()
    agg["prior_team"] = agg["player_id"].map(team_by_weeks)
    agg["opp_unit"] = agg.apply(_opp_unit, axis=1)
    agg["touches"] = agg["carries"].fillna(0) + agg["receptions"].fillna(0)
    agg["yards"] = (agg["rushing_yards"].fillna(0) + agg["receiving_yards"].fillna(0))
    agg["ppr_g"] = agg["ppr"] / agg["games"].clip(lower=1)
    agg["ypt"] = np.where(agg["touches"] > 0, agg["yards"] / agg["touches"], np.nan)
    return agg


def _team_pos_shares(agg: pd.DataFrame, current_team: pd.Series) -> pd.DataFrame:
    """Per (team, position): total workload, and the workload VACATED by players
    who are no longer on that team (departed => their touches are up for grabs)."""
    agg = agg.copy()
    agg["current_team"] = agg["player_id"].map(current_team)
    total = (agg.groupby(["prior_team", "position"])["opp_unit"].sum()
                .rename("team_pos_total").reset_index())
    departed = agg[agg["current_team"].fillna("__gone__") != agg["prior_team"]]
    vac = (departed.groupby(["prior_team", "position"])["opp_unit"].sum()
                   .rename("vacated_units").reset_index())
    m = total.merge(vac, on=["prior_team", "position"], how="left")
    m["vacated_units"] = m["vacated_units"].fillna(0)
    m["team_vacated_share"] = m["vacated_units"] / m["team_pos_total"].clip(lower=1)
    return m


def _recent_momentum(stats: pd.DataFrame) -> pd.DataFrame:
    """In-season signal: last-3-weeks share of position workload vs the player's
    full-season share. >1 = usage trending up. Empty in the preseason."""
    cur = stats[stats["season"] == config.UPCOMING_SEASON].copy()
    if cur.empty:
        return pd.DataFrame(columns=["player_id", "momentum_ratio"])
    cur["opp_unit"] = cur.apply(_opp_unit, axis=1)
    last3 = cur["week"].max() - 2
    recent = cur[cur["week"] >= last3]

    def share(frame):
        tot = frame.groupby(["team", "position"])["opp_unit"].transform("sum").clip(lower=1)
        frame = frame.assign(_share=frame["opp_unit"] / tot)
        return frame.groupby("player_id")["_share"].mean()

    full_s, recent_s = share(cur), share(recent)
    out = pd.DataFrame({"season_share": full_s, "recent_share": recent_s}).reset_index()
    out["momentum_ratio"] = out["recent_share"] / out["season_share"].clip(lower=0.001)
    return out[["player_id", "momentum_ratio"]]


# Roster statuses that mean "not currently a real breakout candidate".
DROP_STATUS = {"RET", "CUT", "TRD", "TRC"}


def build(force: bool = False) -> pd.DataFrame:
    stats = nflverse.weekly_stats(force=force)
    roster = nflverse.roster(force=force)          # tries the upcoming season first
    depth = nflverse.depth_charts(force=force)
    sl = sleeper.players(force=force)
    trend = sleeper.trending(kind="add", force=force)

    # ---- universe = players actually on an NFL roster right now ---------------
    r = roster.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").copy()
    r = r[r["position"].isin(config.SKILL_POSITIONS)]
    r = r[~r["status"].isin(DROP_STATUS)]
    cur_team = r.set_index("gsis_id")["team"]

    agg = _prior_aggregate(stats, config.PRIOR_SEASON)
    shares = _team_pos_shares(agg, cur_team)
    momentum = _recent_momentum(stats)

    df = r[["gsis_id", "full_name", "position", "team", "status", "birth_date",
            "years_exp", "rookie_year", "entry_year", "draft_number", "sleeper_id"]].copy()
    df = df.rename(columns={"full_name": "name"})

    # ---- age / experience ----------------------------------------------------
    bd = pd.to_datetime(df["birth_date"], errors="coerce")
    df["age"] = (SEASON_START - bd).dt.days / 365.25
    df["rookie_season"] = df["rookie_year"].fillna(df["entry_year"]).fillna(config.UPCOMING_SEASON)
    df["exp"] = (config.UPCOMING_SEASON - df["rookie_season"]).clip(lower=0)
    df["is_rookie"] = df["exp"] <= 0

    # Preseason roster releases ship rookies with a null birth_date, so their age
    # comes out NaN. Backfill from the draft-pick record (age at draft, keyed by
    # season + overall pick), advanced to the upcoming season. Uses the raw pick
    # number so undrafted players — who have no pick — never false-match.
    raw_pick = pd.to_numeric(df["draft_number"], errors="coerce")
    if df["age"].isna().any() and raw_pick.notna().any():
        dp = nflverse.draft_picks(force=force)[["season", "pick", "age"]].copy()
        dp = dp.dropna(subset=["pick", "age"]).drop_duplicates(["season", "pick"])
        dp["season"] = dp["season"].astype("Int64")
        dp["pick"] = dp["pick"].astype("Int64")
        key = pd.DataFrame({"season": df["rookie_season"].astype("Int64"),
                            "pick": raw_pick.astype("Int64")})
        draft_age = key.merge(dp, on=["season", "pick"], how="left")["age"]
        season_age = draft_age.to_numpy() + (config.UPCOMING_SEASON - df["rookie_season"].to_numpy())
        df["age"] = df["age"].fillna(pd.Series(season_age, index=df.index))

    # ---- draft capital (overall pick; undrafted -> just past draft end) -------
    df["draft_pick"] = pd.to_numeric(df["draft_number"], errors="coerce").fillna(262)

    # ---- prior-season production --------------------------------------------
    keep = ["player_id", "games", "opp_unit", "touches", "targets", "ppr", "ppr_g",
            "ypt", "wopr", "target_share", "rush_epa", "rec_epa", "prior_team"]
    df = df.merge(agg[keep].rename(columns={"player_id": "gsis_id"}),
                  on="gsis_id", how="left")
    for c in ["games", "opp_unit", "touches", "targets", "ppr", "ppr_g"]:
        df[c] = df[c].fillna(0)

    # personal share of his (prior) team's positional workload
    tot_lookup = shares.set_index(["prior_team", "position"])["team_pos_total"]
    df["team_pos_total"] = df.set_index(["prior_team", "position"]).index.map(tot_lookup)
    df["share_pos"] = df["opp_unit"] / df["team_pos_total"].clip(lower=1)
    df["share_pos"] = df["share_pos"].fillna(0).clip(0, 1)

    # vacated opportunity on his CURRENT team+position
    vac_lookup = shares.set_index(["prior_team", "position"])["team_vacated_share"]
    df["team_vacated_share"] = df.set_index(["team", "position"]).index.map(vac_lookup)
    df["team_vacated_share"] = df["team_vacated_share"].fillna(0).clip(0, 1)

    # ---- depth-chart rank ----------------------------------------------------
    if not depth.empty and "pos_rank" in depth:
        d = depth.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
        df["pos_rank"] = df["gsis_id"].map(d.set_index("gsis_id")["pos_rank"])
    else:
        df["pos_rank"] = np.nan

    # ---- market value + live trending (Sleeper) ------------------------------
    # Coverage is best via sleeper_id (from the roster); fall back to gsis_id,
    # then to a full-name key, then to a looser first-initial+lastname+position
    # key. Rookies carry no sleeper_id on the roster and Sleeper hasn't linked
    # their gsis_id yet, so without the loose key a name variant (Sleeper's
    # "Matt Hibner" vs the roster's "Matthew Hibner") silently drops both their
    # market value AND their trending-adds count to zero.
    def _key(s):
        return s.astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)

    def _loose_key(name: pd.Series, pos: pd.Series) -> pd.Series:
        parts = name.astype(str).str.lower().str.replace(r"[^a-z ]", "", regex=True).str.split()
        first_i = parts.str[0].str[:1].fillna("")
        last = parts.str[-1].fillna("")
        return first_i + last + pos.astype(str).str.lower()

    by_sid = sl.dropna(subset=["sleeper_id"]).drop_duplicates("sleeper_id").set_index("sleeper_id")
    by_gsis = sl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    sl_nm = sl.dropna(subset=["sleeper_name"]).copy()
    sl_nm["_k"] = _key(sl_nm["sleeper_name"])
    by_name = sl_nm.drop_duplicates("_k").set_index("_k")
    sl_nm["_lk"] = _loose_key(sl_nm["sleeper_name"], sl_nm["sleeper_pos"])
    by_loose = sl_nm[sl_nm["_lk"] != ""].drop_duplicates("_lk").set_index("_lk")

    df["_k"] = _key(df["name"])
    df["_lk"] = _loose_key(df["name"], df["position"])
    # fill sleeper_id from gsis, then exact name, then loose name+position
    df["sleeper_id"] = (df["sleeper_id"]
                        .fillna(df["gsis_id"].map(by_gsis["sleeper_id"]))
                        .fillna(df["_k"].map(by_name["sleeper_id"]))
                        .fillna(df["_lk"].map(by_loose["sleeper_id"])))
    # market value (search_rank): sleeper_id -> gsis -> name -> loose
    df["search_rank"] = (df["sleeper_id"].map(by_sid["search_rank"])
                         .fillna(df["gsis_id"].map(by_gsis["search_rank"]))
                         .fillna(df["_k"].map(by_name["search_rank"]))
                         .fillna(df["_lk"].map(by_loose["search_rank"])))
    df["search_rank"] = pd.to_numeric(df["search_rank"], errors="coerce").fillna(10_000_000)
    # Join adds on a normalized string key — roster ids and Sleeper's trending
    # ids must share a dtype or the merge silently misses.
    df["sleeper_id"] = df["sleeper_id"].astype("string")
    trend = trend.copy()
    trend["sleeper_id"] = trend["sleeper_id"].astype("string")
    df = df.merge(trend, on="sleeper_id", how="left")
    df["trend_count"] = df["trend_count"].fillna(0)
    df = df.drop(columns=["_k", "_lk"])

    # ---- momentum ------------------------------------------------------------
    df = df.merge(momentum.rename(columns={"player_id": "gsis_id"}), on="gsis_id", how="left")
    df["momentum_ratio"] = df["momentum_ratio"].fillna(1.0)

    df = df.drop_duplicates("gsis_id").reset_index(drop=True)
    return df


if __name__ == "__main__":
    out = build()
    print(f"Built {len(out)} players")
    print(out[["name", "position", "team", "age", "exp", "share_pos",
               "team_vacated_share", "search_rank"]].head(15).to_string())
