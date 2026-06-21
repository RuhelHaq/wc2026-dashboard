# FIFA World Cup 2026 — Prediction Dashboard

A machine learning system that predicts outcomes for the FIFA World Cup 2026, covering all 48 teams from group stage through the final. Built end-to-end: data collection → feature engineering → model training → tournament simulation → live interactive dashboard.

**🔗 Live dashboard:** [[Streamlit link](https://wc2026-dashboard-hbcy8em7hyao3ncnkj4sjz.streamlit.app)]
**📝 Full writeup:** [[Medium article link](https://medium.com/@ruhelhaqnyc/i-built-a-machine-learning-model-that-predicts-the-fifa-world-cup-2026-heres-everything-that-672a589124cb?postPublishedType=repub)]

---

## Overview

| | |
|---|---|
| **Model** | Poisson GLM + XGBoost ensemble (70/30 weighting) |
| **Validation** | RPS = 0.1670 (vs. ~0.25 for equal-probability baseline) |
| **Simulation** | 10,000 Monte Carlo runs of the full tournament |
| **Dashboard** | Streamlit, 5 pages, deployed on Streamlit Community Cloud |

## How It Works

**1. Data Collection**
FIFA rankings, Elo ratings (calculated from historical matches, weighted by tournament importance), Transfermarkt squad values, historical match results, and a venue heat stress index for the USA/Canada/Mexico host cities.

**2. Feature Engineering**
Elo rating differential, squad value differential, home/away form (calculated separately), home advantage, heat tolerance.

**3. Modeling**
Two Poisson GLMs (`statsmodels`) predict home and away expected goals independently. An XGBoost model trained on the same features captures nonlinear interactions the GLM misses. Final predictions are a weighted ensemble of both.

**4. Simulation**
Each of the 10,000 simulations samples scorelines from the Poisson distributions for every match, advances the correct teams through the group stage and knockout bracket, and tracks the eventual tournament winner. Win probability per team = frequency of winning across all 10,000 runs.

**5. Dashboard**
Five interactive pages — Tournament Overview, Road to the Final, Group Stage, Team Profile, and a Head-to-Head Predictor where you can simulate any matchup on demand.

## Repo Structure

```
├── dashboard.py                          # Live dashboard (Streamlit)
├── requirements.txt                      # Dependencies for deployment
├── Notebook_1_FIFA_WC_26_predictions.ipynb   # Data collection
├── Notebook_2_FIFA_WC_26_Predictions.ipynb   # Feature engineering
├── Notebook_3_FIFA_WC_26_Predictions.ipynb   # Poisson GLM
├── Notebook_4_FIFA_26_prediction.ipynb       # XGBoost + ensemble
├── Notebook_5_FIFA_WC_26_predictions.ipynb   # Monte Carlo simulation
├── Notebook_6_FIFA_WC_26_Prediction.ipynb    # Dashboard generation
├── data/processed/                       # Cleaned datasets, simulation results
└── models/                               # Trained GLM, XGBoost, utils
```

## Key Finding

Feature importance (XGBoost gain) shows **Elo rating differential** dominates both the home and away goal models by a wide margin — well ahead of squad value, form, and home advantage. A direct comparison of USA vs. Mexico's underlying numbers (Mexico: lower squad value, higher Elo, ~8x higher win probability) confirms the model weights actual match performance over market spending. Full breakdown, including where the model's assumptions break down, in the [Medium writeup](your-medium-link).

## Honest Limitations

This isn't built to outpredict bookmakers — it doesn't have access to real-time injury news, lineup leaks, or market-derived signals. It's a transparent, from-scratch statistical approach using public data. With more time, the highest-value additions would be a Dixon-Coles correlation correction, time-decayed Elo weighting, and historical betting odds as a feature.

## Tech Stack

Python · Streamlit · XGBoost · statsmodels · Plotly · pandas

---

*Data cutoff: June 10, 2026. Built by [Ruhel Haq](your-linkedin-link).*
