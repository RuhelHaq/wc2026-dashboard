import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import sys
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import poisson
import statsmodels.api as sm

st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stMainMenu"],
[data-testid="stHeader"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebar"],
.stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.5); border-radius: 3px; }

.stApp {
    background-image: url("https://images.pexels.com/photos/47730/the-ball-stadion-football-the-pitch-47730.jpeg?auto=compress&cs=tinysrgb&w=1200");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
section[data-testid="stMain"] { background-color: rgba(0,10,40,0.84); }

.main .block-container { padding: 1.2rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* Top nav bar styling */
.topnav-wrap {
    background: rgba(0,8,32,0.85);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.topnav-wrap [data-testid="stRadio"] > label { display: none !important; }
.topnav-wrap [data-testid="stRadio"] > div { gap: 4px !important; flex-direction: row !important; }
.topnav-wrap [data-testid="stRadio"] div[role="radio"] {
    background: transparent !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 6px 14px !important;
}
.topnav-wrap [data-testid="stRadio"] div[role="radio"]:hover {
    background: rgba(201,168,76,0.08) !important;
}
.topnav-wrap [data-testid="stRadio"] div[role="radio"][aria-checked="true"] {
    background: rgba(201,168,76,0.14) !important;
    border-bottom: 3px solid #C9A84C !important;
}
.topnav-wrap [data-testid="stRadio"] div[role="radio"] p {
    font-size: 0.92rem !important;
    color: rgba(255,255,255,0.6) !important;
    margin: 0 !important;
    white-space: nowrap;
}
.topnav-wrap [data-testid="stRadio"] div[role="radio"][aria-checked="true"] p {
    color: #C9A84C !important;
    font-weight: 700 !important;
}
.topnav-wrap [data-testid="stRadio"] div[role="radio"] > div:first-child { display: none !important; }

h1 { color: #ffffff !important; font-weight: 700 !important; }
h2 { color: #C9A84C !important; font-weight: 600 !important; }
h3 { color: #C9A84C !important; font-weight: 600 !important; }
p, li { color: rgba(255,255,255,0.85) !important; }

[data-testid="metric-container"] {
    background-color: rgba(0,20,70,0.75) !important;
    border: 1px solid rgba(201,168,76,0.45) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
}
[data-testid="metric-container"] * { color: white !important; }
[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] { color: #fff !important; font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #C9A84C !important; }

[data-testid="stSelectbox"] > div > div {
    background: rgba(0,15,55,0.85) !important;
    border: 1px solid rgba(201,168,76,0.35) !important;
    border-radius: 8px !important;
    color: white !important;
}
[data-testid="stSelectbox"] label {
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stCheckbox"] label { color: rgba(255,255,255,0.7) !important; }
hr { border-color: rgba(201,168,76,0.25) !important; }

/* Selectbox dropdown popover (the list that opens) */
div[data-baseweb="popover"] {
    background: #0a1230 !important;
}
div[data-baseweb="popover"] ul {
    background: #0a1230 !important;
}
div[data-baseweb="popover"] li {
    background: #0a1230 !important;
    color: rgba(255,255,255,0.85) !important;
}
div[data-baseweb="popover"] li:hover {
    background: rgba(201,168,76,0.15) !important;
    color: #C9A84C !important;
}
div[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(201,168,76,0.2) !important;
    color: #C9A84C !important;
}

/* Dark theme dataframe - target the actual canvas/grid cells */
div[data-testid="stDataFrame"] {
    background: rgba(0,12,45,0.9) !important;
    border: 1px solid rgba(201,168,76,0.25) !important;
    border-radius: 8px !important;
}
div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
    background: rgba(0,12,45,0.9) !important;
}
div[data-testid="stDataFrame"] .glideDataEditor,
div[data-testid="stDataFrame"] canvas {
    background: rgba(0,12,45,0.9) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
processed_dir = BASE_DIR / "data" / "processed"
models_dir    = BASE_DIR / "models"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    results         = pd.read_csv(processed_dir / "simulation_results.csv")
    team_features   = pd.read_csv(processed_dir / "team_features_2026.csv", index_col="team")
    fixture_lambdas = pd.read_csv(processed_dir / "fixture_lambdas.csv")
    fixtures        = pd.read_csv(processed_dir / "fixtures_2026_clean.csv", parse_dates=["date"])
    with open(processed_dir / "group_info.pkl", "rb") as f:
        group_info = pickle.load(f)
    return results, team_features, fixture_lambdas, fixtures, group_info

@st.cache_resource
def load_models():
    with open(models_dir / "glm_home_v2.pkl", "rb") as f:
        glm_home = pickle.load(f)
    with open(models_dir / "glm_away_v2.pkl", "rb") as f:
        glm_away = pickle.load(f)
    xgb_home = xgb.Booster()
    xgb_home.load_model(str(models_dir / "xgb_home.json"))
    xgb_away = xgb.Booster()
    xgb_away.load_model(str(models_dir / "xgb_away.json"))
    with open(models_dir / "ensemble_alpha.pkl", "rb") as f:
        alpha = pickle.load(f)
    with open(models_dir / "features_v2.pkl", "rb") as f:
        features = pickle.load(f)
    sys.path.insert(0, str(models_dir))
    from utils import match_probabilities
    return glm_home, glm_away, xgb_home, xgb_away, alpha, features, match_probabilities

results, team_features, fixture_lambdas, fixtures, group_info = load_data()
glm_home, glm_away, xgb_home, xgb_away, BEST_ALPHA, FEATURES_V2, match_probabilities = load_models()

# ── Predict lambda ─────────────────────────────────────────────────────────────
def predict_lambda(home_team, away_team, home_advantage=0, home_form=0.5, away_form=0.5):
    home = team_features.loc[home_team]
    away = team_features.loc[away_team]
    feat = pd.DataFrame([{
        "elo_diff":         home["elo_rating"]        - away["elo_rating"],
        "squad_value_diff": home["squad_value_eur_m"] - away["squad_value_eur_m"],
        "home_form":        home_form,
        "away_form":        away_form,
        "home_advantage":   home_advantage,
    }])
    X      = sm.add_constant(feat[FEATURES_V2], has_constant="add")
    lh_glm = float(glm_home.predict(X).iloc[0])
    la_glm = float(glm_away.predict(X).iloc[0])
    dmat   = xgb.DMatrix(feat[FEATURES_V2])
    lh_xgb = float(xgb_home.predict(dmat)[0])
    la_xgb = float(xgb_away.predict(dmat)[0])
    lh = BEST_ALPHA * lh_glm + (1 - BEST_ALPHA) * lh_xgb
    la = BEST_ALPHA * la_glm + (1 - BEST_ALPHA) * la_xgb
    return min(lh, 6.0), min(la, 5.0)

# ── Plotly theme ──────────────────────────────────────────────────────────────
GOLD = "#C9A84C"

def base_layout(height=400):
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter, sans-serif", size=13),
        height=height,
        margin=dict(l=10, r=20, t=10, b=40),
        xaxis=dict(color="white", tickfont=dict(size=12, color="rgba(255,255,255,0.75)"),
                   gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(color="white", tickfont=dict(size=12, color="rgba(255,255,255,0.75)"),
                   gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
        showlegend=False,
    )

def gold_bars(n, top_n=3):
    colors = []
    for i in range(n):
        rank  = n - 1 - i
        alpha = round(1.0 - rank * 0.12, 2) if rank < top_n else round(max(0.25, 0.72 - (rank - top_n) * 0.04), 2)
        colors.append(f"rgba(201,168,76,{alpha})")
    return colors

def dark_table(df):
    """Render a DataFrame as a plain dark-themed HTML table (no virtualized scrolling, no white-on-scroll bug)."""
    headers = "".join(f'<th>{c}</th>' for c in df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        cells = "".join(f'<td>{v}</td>' for v in row)
        rows_html += f"<tr>{cells}</tr>"
    html = f"""
    <div style="max-height:680px;overflow-y:auto;border:1px solid rgba(201,168,76,0.25);border-radius:8px;">
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>
    <style>
    table th {{
        background:#070d24 !important; color:#C9A84C !important; font-weight:700 !important;
        text-transform:uppercase; font-size:0.7rem; letter-spacing:0.05em;
        padding:8px 10px; text-align:right; border-bottom:1px solid rgba(201,168,76,0.25);
        position: sticky; top: 0; z-index: 1;
    }}
    table th:first-child {{ text-align:left; }}
    table td {{
        background:#0a1230 !important; color:#fff !important;
        padding:6px 10px; text-align:right; border-bottom:1px solid rgba(255,255,255,0.05);
        font-size:0.85rem;
    }}
    table td:first-child {{ text-align:left; font-weight:500; }}
    table tr:hover td {{ background: rgba(201,168,76,0.08) !important; }}
    </style>
    """
    return html

def show_table(df, height=None):
    st.markdown(dark_table(df), unsafe_allow_html=True)

# ── Top navigation bar ──────────────────────────────────────────────────────────
st.markdown('<div class="topnav-wrap">', unsafe_allow_html=True)
logo_col, nav_col, info_col = st.columns([1, 6, 2])
with logo_col:
    st.image(
        "https://assets.football-logos.cc/logos/tournaments/1500x1500/fifa-world-cup-2026.31d2489d.png",
        width=56,
    )
with nav_col:
    _page = st.radio("Navigate", [
        "🏆  Overview", "🛣️  Road to Final", "👥  Group Stage",
        "👤  Team Profile", "⚔️  Head to Head",
    ], horizontal=True, label_visibility="collapsed")
    page = _page.split("  ", 1)[1]
with info_col:
    st.markdown(
        '<div style="text-align:right;font-size:0.7rem;color:rgba(255,255,255,0.4);line-height:1.5;padding-top:4px;">'
        '<span style="color:#C9A84C;font-weight:700;">Built by Ruhel Haq</span><br>'
        'Poisson GLM + XGBoost · 10K Monte Carlo sims<br>'
        'Elo ratings · squad values · heat stress</div>',
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

# Map nav labels back to full page names used below
page_map = {
    "Overview": "Tournament Overview",
    "Road to Final": "Road to the Final",
    "Group Stage": "Group Stage",
    "Team Profile": "Team Profile",
    "Head to Head": "Head to Head",
}
page = page_map.get(page, page)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Tournament Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Tournament Overview":
    st.title("FIFA World Cup 2026 — Tournament Predictions")
    st.caption("Probabilities based on 10,000 Monte Carlo simulations · Elo ratings · squad values · historical match data")

    with st.expander("ℹ️ About this model"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            **🤖 Machine Learning Model**
            
            Ensemble of Poisson GLM and XGBoost models trained on historical international match data. 
            Goals scored by each team are modelled independently using a Poisson distribution.
            
            Model performance: **RPS = 0.1670**
            """)
        with col_b:
            st.markdown("""
            **⚙️ Key Features**
            
            - Elo rating difference between teams
            - Squad market value differential (Transfermarkt)
            - Recent form (home & away separately)
            - Home advantage factor
            - Venue heat stress index
            """)
        with col_c:
            st.markdown("""
            **🎲 Simulation**
            
            10,000 Monte Carlo simulations of the full tournament — group stage through the final. 
            Each simulation samples scorelines from the Poisson model and tracks every team's path.
            
            Data cutoff: **June 10, 2026**
            """)

    fav = results.nlargest(1, "Winner").iloc[0]
    co  = results.nlargest(2, "Winner").iloc[1]

    c1, c2, c3 = st.columns(3)
    c1.metric("🥇 Tournament Favourite", fav["Team"], f"↑ {fav['Winner']:.1f}% win probability")
    c2.metric("🥈 Co-Favourite",          co["Team"],  f"↑ {co['Winner']:.1f}% win probability")
    c3.metric("🎲 Simulations Run",       "10,000",    "↑ 376 seconds")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_table = st.columns([60, 40])

    with col_chart:
        st.subheader("Winner Probabilities — All 48 Teams")
        plot_df = results.sort_values("Winner").copy()
        n       = len(plot_df)
        fig = go.Figure(go.Bar(
            x=plot_df["Winner"], y=plot_df["Team"], orientation="h",
            marker=dict(color=gold_bars(n, top_n=3), line=dict(width=0)),
            text=plot_df["Winner"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.55)", size=10),
        ))
        layout = base_layout(height=max(700, n * 24))
        layout["xaxis"]["title"] = dict(text="Win Probability (%)", font=dict(size=12))
        layout["yaxis"]["tickfont"] = dict(size=11, color="rgba(255,255,255,0.8)")
        layout["margin"] = dict(l=120, r=60, t=10, b=40)
        fig.update_layout(**layout)
        st.plotly_chart(fig, width='stretch')

    with col_table:
        st.subheader("Full Probability Table")
        stage_cols = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Winner"]
        disp = results[["Team"] + [c for c in stage_cols if c in results.columns]].copy()
        disp = disp.sort_values("Winner", ascending=False).reset_index(drop=True)
        disp.index += 1
        disp.columns = [c if c == "Team" else
                         {"Round of 32": "R32", "Round of 16": "R16", "Quarter-final": "QF",
                          "Semi-final": "SF", "Final": "Final", "Winner": "Win"}.get(c, c)
                         for c in disp.columns]
        for col in disp.columns[1:]:
            disp[col] = disp[col].apply(lambda x: f"{x:.1f}%")
        show_table(disp)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Road to the Final
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Road to the Final":
    st.title("Road to the Final")
    st.caption("Which teams are most likely to go deep into the tournament?")
    st.markdown("---")

    st.subheader("Top 8 — Semi-final Contenders")
    top8 = results.nlargest(8, "Semi-final")[["Team", "Semi-final", "Final", "Winner"]].reset_index(drop=True)

    fig1 = go.Figure(go.Bar(
        x=top8["Team"], y=top8["Semi-final"],
        marker=dict(color=list(reversed(gold_bars(8, top_n=2))), line=dict(width=0)),
        text=top8["Semi-final"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="rgba(255,255,255,0.6)", size=12),
    ))
    l1 = base_layout(height=340)
    l1["yaxis"]["title"] = dict(text="Semi-final Probability (%)", font=dict(size=12))
    l1["margin"] = dict(l=20, r=20, t=10, b=50)
    fig1.update_layout(**l1)
    st.plotly_chart(fig1, width='stretch')

    cols8 = st.columns(8)
    for col, row in zip(cols8, top8.itertuples()):
        col.metric(row.Team, f"{row._2:.1f}%")

    st.markdown("---")
    st.subheader("Top 4 — Final Contenders")
    top4 = results.nlargest(4, "Final")[["Team", "Final", "Winner"]].reset_index(drop=True)
    cols4 = st.columns(4)
    for col, row in zip(cols4, top4.itertuples()):
        col.metric(row.Team, f"{row.Final:.1f}%", f"Win: {row.Winner:.1f}%")

    st.markdown("---")
    st.subheader("Most Likely Champion")
    winner_row = results.nlargest(1, "Winner").iloc[0]
    top10      = results.nlargest(10, "Winner")[["Team", "Winner"]].reset_index(drop=True)

    col_champ, col_chart2 = st.columns([1, 2])
    with col_champ:
        st.metric("🏆 " + winner_row["Team"], f"{winner_row['Winner']:.1f}%", "Win probability")
    with col_chart2:
        fig2 = go.Figure(go.Bar(
            x=top10["Team"], y=top10["Winner"],
            marker=dict(color=list(reversed(gold_bars(len(top10), top_n=1))), line=dict(width=0)),
            text=top10["Winner"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside", textfont=dict(color="rgba(255,255,255,0.6)", size=12),
        ))
        l2 = base_layout(height=300)
        l2["yaxis"]["title"] = dict(text="Win Probability (%)", font=dict(size=12))
        l2["margin"] = dict(l=10, r=20, t=10, b=60)
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Group Stage
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Group Stage":
    st.title("Group Stage")
    selected_group = st.selectbox("Select Group", sorted(group_info.keys()))
    group_teams    = group_info[selected_group]
    group_fixtures = fixture_lambdas[fixture_lambdas["group"] == selected_group].copy()

    st.subheader(f"Group {selected_group} — Teams")
    rows = []
    for team in group_teams:
        tf  = team_features.loc[team]
        res = results[results["Team"] == team].iloc[0]
        rows.append({
            "Team": team, "Elo Rating": int(tf["elo_rating"]), "FIFA Rank": int(tf["fifa_rank"]),
            "Squad Value": f"€{int(tf['squad_value_eur_m'])}M",
            "Qualify %": f"{res['Round of 32']:.1f}%", "Win %": f"{res['Winner']:.1f}%",
        })
    group_df = pd.DataFrame(rows).sort_values("Elo Rating", ascending=False).reset_index(drop=True)
    group_df.index += 1
    show_table(group_df)

    st.subheader(f"Group {selected_group} — Fixtures")
    for _, row in group_fixtures.iterrows():
        if pd.notna(row["lh"]):
            probs = match_probabilities(row["lh"], row["la"])
            c1, c2, c3 = st.columns([2, 3, 2])
            c1.markdown(f'<div style="text-align:right;padding:0.5rem 0;"><b style="color:white;font-size:0.95rem;">{row["home"]}</b></div>', unsafe_allow_html=True)
            c2.markdown(
                f'<div style="text-align:center;padding:0.5rem 0;">'
                f'<span style="color:rgba(255,255,255,0.75);font-size:0.85rem;">xG: {row["lh"]:.2f} — {row["la"]:.2f}</span><br>'
                f'<span style="color:rgba(255,255,255,0.42);font-size:0.72rem;">'
                f'{row["home"]} win {probs["home_win"]:.0%} | Draw {probs["draw"]:.0%} | {row["away"]} win {probs["away_win"]:.0%}</span></div>',
                unsafe_allow_html=True,
            )
            c3.markdown(f'<div style="text-align:left;padding:0.5rem 0;"><b style="color:white;font-size:0.95rem;">{row["away"]}</b></div>', unsafe_allow_html=True)
            st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Team Profile
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Team Profile":
    st.title("Team Profile")
    all_teams     = sorted(results["Team"].tolist())
    selected_team = st.selectbox("Select Team", all_teams)

    tf  = team_features.loc[selected_team]
    res = results[results["Team"] == selected_team].iloc[0]

    st.subheader(selected_team)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elo Rating",      f"{tf['elo_rating']:.0f}")
    c2.metric("FIFA Rank",       f"#{int(tf['fifa_rank'])}")
    c3.metric("Squad Value",     f"€{tf['squad_value_eur_m']:.0f}M")
    c4.metric("Win Probability", f"{res['Winner']:.1f}%")

    st.markdown("---")
    st.subheader("Tournament Stage Probabilities")
    stages     = [s for s in ["Round of 32","Round of 16","Quarter-final","Semi-final","Final","Winner"] if s in res.index]
    probs_list = [res[s] for s in stages]
    max_p      = max(probs_list)
    bar_cols   = [GOLD if p == max_p else f"rgba(201,168,76,{round(max(0.28, 0.85 - (max_p - p) * 0.018), 2)})" for p in probs_list]

    fig_t = go.Figure(go.Bar(
        x=stages, y=probs_list, marker=dict(color=bar_cols, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs_list], textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.65)", size=13),
    ))
    lt = base_layout(height=320)
    lt["yaxis"]["title"] = dict(text="Probability (%)", font=dict(size=12))
    lt["margin"] = dict(l=20, r=20, t=10, b=40)
    fig_t.update_layout(**lt)
    st.plotly_chart(fig_t, width='stretch')

    st.markdown("---")
    st.subheader("Group Fixtures")
    team_fixtures = fixture_lambdas[
        ((fixture_lambdas["home"] == selected_team) | (fixture_lambdas["away"] == selected_team)) &
        (fixture_lambdas["stage"] == "Group")
    ]
    for _, row in team_fixtures.iterrows():
        if pd.notna(row["lh"]):
            is_home  = row["home"] == selected_team
            opponent = row["away"] if is_home else row["home"]
            lh       = row["lh"] if is_home else row["la"]
            la       = row["la"] if is_home else row["lh"]
            probs    = match_probabilities(lh, la)
            st.markdown(
                f'<div style="background:rgba(0,15,55,0.65);border:1px solid rgba(201,168,76,0.2);'
                f'border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;">'
                f'<b style="color:white;">vs {opponent}</b>'
                f'<span style="float:right;color:rgba(255,255,255,0.55);font-size:0.82rem;">'
                f'Win {probs["home_win"]:.1%} | Draw {probs["draw"]:.1%} | Loss {probs["away_win"]:.1%} | xG {lh:.2f}–{la:.2f}</span></div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Head to Head
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Head to Head":
    st.title("Head to Head Predictor")
    st.caption("Select any two teams to see match predictions.")

    all_teams = sorted(results["Team"].tolist())
    c1, c2    = st.columns(2)
    home_team = c1.selectbox("Home Team", all_teams, index=all_teams.index("Argentina"))
    away_team = c2.selectbox("Away Team", all_teams, index=all_teams.index("France"))

    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        home_adv = st.checkbox(f"Home advantage for {home_team}", value=False)
        lh, la   = predict_lambda(home_team, away_team, home_advantage=int(home_adv))
        probs    = match_probabilities(lh, la)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} Win", f"{probs['home_win']*100:.1f}%")
        c2.metric("Draw",             f"{probs['draw']*100:.1f}%")
        c3.metric(f"{away_team} Win", f"{probs['away_win']*100:.1f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric(f"xG — {home_team}", f"{lh:.2f}")
        c2.metric(f"xG — {away_team}", f"{la:.2f}")

        st.markdown("---")
        st.subheader("Top 10 Most Likely Scorelines")

        rows = []
        for i in range(8):
            for j in range(8):
                p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                if i > j:    result = f"{home_team} win"
                elif i == j: result = "Draw"
                else:        result = f"{away_team} win"
                rows.append({"Scoreline": f"{i} – {j}", "Probability %": round(p * 100, 2), "Result": result})

        sc_df = pd.DataFrame(rows).sort_values("Probability %", ascending=False).head(10).reset_index(drop=True)
        sc_df.index += 1
        sc_df["Probability %"] = sc_df["Probability %"].apply(lambda x: f"{x:.2f}%")
        show_table(sc_df)

        st.markdown("---")
        st.subheader("Scoreline Distribution")

        chart_df = pd.DataFrame(rows).sort_values("Probability %", ascending=False).head(10).reset_index(drop=True)
        color_map = {f"{home_team} win": GOLD, "Draw": "rgba(150,150,150,0.65)", f"{away_team} win": "#7db8e8"}
        fig_sc = go.Figure(go.Bar(
            x=chart_df["Scoreline"], y=chart_df["Probability %"],
            marker=dict(color=[color_map.get(r, GOLD) for r in chart_df["Result"]], line=dict(width=0)),
            text=chart_df["Probability %"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside", textfont=dict(color="rgba(255,255,255,0.6)", size=11),
        ))
        lsc = base_layout(height=320)
        lsc["yaxis"]["title"] = dict(text="Probability (%)", font=dict(size=12))
        lsc["margin"] = dict(l=20, r=20, t=10, b=50)
        fig_sc.update_layout(**lsc)
        st.plotly_chart(fig_sc, width='stretch')
