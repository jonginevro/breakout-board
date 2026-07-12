"""Live Breakout Board — Streamlit dashboard.

Run with:  ./run.sh   (or)   .venv/bin/streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a top-level script, so relative imports have no
# parent package. Put the project root on the path and import the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.model.scoring import score_players
from src import config

st.set_page_config(page_title="Fantasy Breakout Board", layout="wide",
                   page_icon="🚀")

FACTORS = ["opportunity", "catalyst", "efficiency", "suppression"]
FACTOR_HELP = {
    "opportunity": "Room for volume to grow (vacated team touches, personal headroom, depth-chart gap).",
    "catalyst": "Likelihood that room opens for HIM (draft capital, age window, 2nd-year leap, depth proximity, manual notes).",
    "efficiency": "Can he convert volume to points (per-touch production, usage quality, talent prior).",
    "suppression": "How cheap-but-on-the-radar he is (value sweet spot). Elite & off-radar players score low.",
}


@st.cache_data(ttl=3600, show_spinner="Crunching breakout scores…")
def load(force: bool = False) -> pd.DataFrame:
    return score_players(force=force)


def factor_bars(row: pd.Series) -> go.Figure:
    vals = [row[f] for f in FACTORS]
    fig = go.Figure(go.Bar(
        x=vals, y=[f.capitalize() for f in FACTORS], orientation="h",
        marker_color=["#4C9BE8", "#E8994C", "#5FBB7D", "#B45FE8"],
        text=[f"{v:.2f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(xaxis_range=[0, 1.05], height=200,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=None, yaxis=dict(autorange="reversed"))
    return fig


# ============================== Sidebar ======================================
st.sidebar.title("Breakout Board")
st.sidebar.caption(f"Projecting the **{config.UPCOMING_SEASON}** season, "
                   f"prior baseline **{config.PRIOR_SEASON}**")

if st.sidebar.button("Refresh live data"):
    load.clear()
    st.session_state["_force"] = True

data = load(force=st.session_state.pop("_force", False))
# "In-season" = we actually have current-season game usage (momentum_ratio moves).
# Sleeper trending adds nudge `momentum` year-round, so don't key the mode off it.
in_season = (data["momentum_ratio"] != 1.0).any()

positions = st.sidebar.multiselect("Positions", config.SKILL_POSITIONS,
                                   default=config.SKILL_POSITIONS)
exp_max = st.sidebar.slider("Max years of experience", 0, 15, 6,
                            help="Breakouts skew young. 0 = rookies only.")
value_band = st.sidebar.select_slider(
    "Value tier (Sleeper search rank)",
    options=["Any", "Startable (≤100)", "Cheap (≤300)", "Deep (≤600)"],
    value="Any")
name_q = st.sidebar.text_input("Find a player").strip().lower()
top_n = st.sidebar.number_input("Show top N", 10, 300, 50, step=10)

# ============================== Filtering ====================================
df = data[data["position"].isin(positions)].copy()
df = df[df["exp"] <= exp_max]
band = {"Startable (≤100)": 100, "Cheap (≤300)": 300, "Deep (≤600)": 600}
if value_band in band:
    df = df[df["search_rank"] <= band[value_band]]
if name_q:
    df = df[df["name"].str.lower().str.contains(name_q)]

# ============================== Header =======================================
st.title("Fantasy Football Breakout Board")
c1, c2, c3 = st.columns(3)
c1.metric("Players scored", f"{len(data):,}")
c2.metric("In candidate view", f"{len(df):,}")
c3.metric("Mode", "In-season" if in_season else "Preseason")

st.caption("Score = geomean(opportunity · catalyst · efficiency) × value sweet-spot"
           + (" × live momentum" if in_season else "")
           + ". The geometric mean means a player must hit on ALL of opportunity, "
             "catalyst and efficiency.")

# ============================== Leaderboard ==================================
show = df.head(int(top_n)).copy()
show.insert(0, "rank", range(1, len(show) + 1))
table_cols = ["rank", "name", "position", "team", "age", "exp", "breakout_score",
              *FACTORS, "momentum", "team_vacated_share", "share_pos", "pos_rank",
              "draft_pick", "ppr_g", "trend_count", "search_rank"]
st.dataframe(
    show[table_cols],
    hide_index=True, width="stretch", height=560,
    column_config={
        "name": "Player",
        "breakout_score": st.column_config.ProgressColumn(
            "Breakout", min_value=0, max_value=float(data["breakout_score"].max()),
            format="%.1f"),
        "age": st.column_config.NumberColumn("Age", format="%.1f"),
        **{f: st.column_config.NumberColumn(f.capitalize()[:4], format="%.2f",
                                            help=FACTOR_HELP[f]) for f in FACTORS},
        "momentum": st.column_config.NumberColumn("Mom", format="%.2f",
                    help="In-season usage trend × trending adds (1.0 = neutral)."),
        "team_vacated_share": st.column_config.NumberColumn("Vacated%", format="%.2f",
                    help="Share of his team's positional workload left by departed players."),
        "share_pos": st.column_config.NumberColumn("UsedShr", format="%.2f",
                    help="His share of the team's positional workload last year."),
        "pos_rank": st.column_config.NumberColumn("Depth", format="%.0f"),
        "draft_pick": st.column_config.NumberColumn("Pick", format="%.0f"),
        "ppr_g": st.column_config.NumberColumn("PPR/G", format="%.1f"),
        "trend_count": st.column_config.NumberColumn("Adds", format="%.0f"),
        "search_rank": st.column_config.NumberColumn("Value", format="%.0f"),
    },
)
st.caption("Tip: click a column header to re-sort. 'Value' is Sleeper search rank "
           "(lower = more valued). Edit config/weights.yaml to retune the model.")

# ============================== Player detail ================================
st.divider()
st.subheader("Why is a player rated this way?")
options = df["name"].tolist()
if options:
    default = "Kaleb Johnson" if "Kaleb Johnson" in options else options[0]
    pick = st.selectbox("Player", options, index=options.index(default))
    row = df[df["name"] == pick].iloc[0]

    left, right = st.columns([1, 1])
    with left:
        st.metric(f"{pick}  ({row['position']} · {row['team']})",
                  f"{row['breakout_score']:.1f}",
                  help="Breakout score")
        st.plotly_chart(factor_bars(row))  # fills its column by default
    with right:
        age_txt = f"{row['age']:.1f} yrs" if pd.notna(row['age']) else "age n/a"
        st.markdown(f"""
- Age / experience: {age_txt} · year {int(row['exp'])+1}
- Draft capital: pick {int(row['draft_pick'])}  ·  depth rank: {row['pos_rank'] if pd.notna(row['pos_rank']) else '—'}
- Vacated workload on team: {row['team_vacated_share']:.0%}  ·  his prior share: {row['share_pos']:.0%}
- Last-year production: {row['ppr_g']:.1f} PPR/g over {int(row['games'])} games
- Market value (Sleeper rank): {int(row['search_rank']) if row['search_rank']<1e6 else 'off-radar'}  ·  live adds: {int(row['trend_count'])}
- Momentum multiplier: {row['momentum']:.2f}
""")
        weak = min(FACTORS[:3], key=lambda f: row[f])
        strong = max(FACTORS[:3], key=lambda f: row[f])
        st.info(f"Read: carried by {strong}({row[strong]:.2f}); "
                f"the ceiling on his score is {weak} ({row[weak]:.2f}). "
                + FACTOR_HELP[weak])
else:
    st.write("No players match the current filters.")
