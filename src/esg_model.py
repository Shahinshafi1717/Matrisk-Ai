# src/esg_model.py
# ESG Scoring Model for Commodities
# Run with: python src/esg_model.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/figures", exist_ok=True)

print("=" * 60)
print("  MatRisk AI - ESG Scoring Model")
print("=" * 60)

# ── STEP 1: Load data ─────────────────────────────────────
print("\n[1] Loading data...")
df4 = pd.read_csv("data/raw/DS4_crossdomain_features_daily.csv",
                  parse_dates=["date"])
df2 = pd.read_csv("data/raw/DS2_commodity_prices_10yr.csv",
                  parse_dates=["date"])
df  = pd.merge(df4, df2[["date","commodity","close"]],
               on=["date","commodity"], how="inner")
print(f"    Rows: {len(df)}")
print(f"    Commodities: {df['commodity'].unique().tolist()}")

# ── STEP 2: Calculate ESG components ─────────────────────
print("\n[2] Calculating ESG scores...")

# Use latest available data per commodity
latest = df.sort_values("date").groupby("commodity").last().reset_index()

# ── ENVIRONMENTAL SCORE (E) ───────────────────────────────
# Lower carbon intensity = better E score
# Higher recyclability = better E score

# Carbon reduction potential (virgin vs recycled)
latest["carbon_reduction"] = (
    (latest["carbon_intensity_virgin"] -
     latest["carbon_intensity_recycled"]) /
    latest["carbon_intensity_virgin"]
)

# Normalize carbon intensity (lower = better = higher score)
max_carbon = latest["carbon_intensity_virgin"].max()
latest["e_carbon_score"] = (
    1 - latest["carbon_intensity_virgin"] / max_carbon
) * 100

# Green premium shows green energy demand
max_green = latest["green_premium_per_kg"].max()
latest["e_green_score"] = (
    latest["green_premium_per_kg"] / max_green
) * 100

# Environmental score = average of both
latest["E_score"] = (
    latest["e_carbon_score"] * 0.6 +
    latest["e_green_score"] * 0.4
).round(1)

# ── SOCIAL SCORE (S) ─────────────────────────────────────
# Lower supply disruption = better S score
# Lower market concentration = better S score (more fair)

max_disruption = latest["supply_disruption_prob"].max()
latest["s_supply_score"] = (
    1 - latest["supply_disruption_prob"] / max_disruption
) * 100

# Herfindahl index measures monopoly risk
# High HHI = market dominated by few suppliers = bad
max_hhi = latest["herfindahl_index"].max()
latest["s_market_score"] = (
    1 - latest["herfindahl_index"] / max_hhi
) * 100

latest["S_score"] = (
    latest["s_supply_score"] * 0.5 +
    latest["s_market_score"] * 0.5
).round(1)

# ── GOVERNANCE SCORE (G) ─────────────────────────────────
# Substitution elasticity = can we switch to alternatives?
# Higher substitutability = better governance resilience

max_sub = latest["substitution_elasticity"].max()
latest["G_score"] = (
    latest["substitution_elasticity"] / max_sub * 100
).round(1)

# ── COMBINED ESG SCORE ────────────────────────────────────
latest["ESG_score"] = (
    latest["E_score"] * 0.40 +
    latest["S_score"] * 0.35 +
    latest["G_score"] * 0.25
).round(1)

# ESG rating
def esg_rating(score):
    if score >= 70:
        return "AAA"
    elif score >= 55:
        return "AA"
    elif score >= 40:
        return "A"
    elif score >= 25:
        return "BBB"
    else:
        return "BB"

latest["ESG_rating"] = latest["ESG_score"].apply(esg_rating)

# ── STEP 3: Print results ─────────────────────────────────
print("\n[3] ESG Scores by Commodity:")
print(f"\n    {'Commodity':<20} {'E':>6} {'S':>6} {'G':>6} "
      f"{'ESG':>6} {'Rating':>8}")
print(f"    {'-'*55}")

results = latest[["commodity","E_score","S_score","G_score",
                   "ESG_score","ESG_rating"]].sort_values(
    "ESG_score", ascending=False)

for _, row in results.iterrows():
    bar = "█" * int(row["ESG_score"] / 5)
    print(f"    {row['commodity']:<20} "
          f"{row['E_score']:>6.1f} "
          f"{row['S_score']:>6.1f} "
          f"{row['G_score']:>6.1f} "
          f"{row['ESG_score']:>6.1f} "
          f"{row['ESG_rating']:>8}")

# ── STEP 4: Carbon analysis ───────────────────────────────
print("\n[4] Carbon reduction potential (virgin vs recycled):")
carbon = latest[["commodity","carbon_intensity_virgin",
                 "carbon_intensity_recycled",
                 "carbon_reduction"]].sort_values(
    "carbon_reduction", ascending=False)

for _, row in carbon.iterrows():
    bar = "█" * int(row["carbon_reduction"] * 20)
    print(f"    {row['commodity']:<20} "
          f"virgin={row['carbon_intensity_virgin']:>6.1f}  "
          f"recycled={row['carbon_intensity_recycled']:>6.1f}  "
          f"saving={row['carbon_reduction']:>5.1%}  "
          f"{bar}")

# ── STEP 5: ESG trend over time ───────────────────────────
print("\n[5] Computing ESG trends over time...")

def compute_esg_row(row):
    return (row["mqi"] * 0.4 +
            (1 - row["supply_disruption_prob"]) * 35 +
            row["substitution_elasticity"] * 0.25)

df["esg_proxy"] = (
    df["mqi"] * 0.40 +
    (1 - df["supply_disruption_prob"]) * 35 +
    df["substitution_elasticity"] * 0.25
)

# ── STEP 6: Save charts ───────────────────────────────────
print("\n[6] Saving charts...")

# Chart 1: ESG scores radar-style bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x      = np.arange(len(results))
width  = 0.25
bars_e = ax.bar(x - width, results["E_score"],
                width, label="E (Environment)", color="#2ca02c")
bars_s = ax.bar(x,         results["S_score"],
                width, label="S (Social)",      color="#1f77b4")
bars_g = ax.bar(x + width, results["G_score"],
                width, label="G (Governance)",  color="#9467bd")
ax.set_title("ESG Component Scores by Commodity")
ax.set_xlabel("Commodity")
ax.set_ylabel("Score (0-100)")
ax.set_xticks(x)
ax.set_xticklabels(results["commodity"],
                   rotation=45, ha="right")
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/esg_component_scores.png", dpi=130)
plt.close()
print("    Saved: esg_component_scores.png")

# Chart 2: Overall ESG ranking
fig, ax = plt.subplots(figsize=(10, 6))
esg_colors = {"AAA":"#2ca02c","AA":"#55A868",
              "A":"#ff7f0e","BBB":"#d62728","BB":"#8B0000"}
bar_colors = [esg_colors.get(r,"#999")
              for r in results["ESG_rating"]]
bars = ax.barh(results["commodity"],
               results["ESG_score"],
               color=bar_colors, edgecolor="white")
ax.set_title("Overall ESG Score Ranking by Commodity")
ax.set_xlabel("ESG Score (0-100)")
ax.set_xlim(0, 100)
for bar, (_, row) in zip(bars, results.iterrows()):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{row['ESG_score']:.1f} ({row['ESG_rating']})",
            va="center", fontsize=10)
plt.tight_layout()
plt.savefig("notebooks/figures/esg_overall_ranking.png", dpi=130)
plt.close()
print("    Saved: esg_overall_ranking.png")

# Chart 3: Carbon intensity comparison
fig, ax = plt.subplots(figsize=(12, 5))
x2     = np.arange(len(carbon))
width2 = 0.35
ax.bar(x2 - width2/2,
       carbon["carbon_intensity_virgin"],
       width2, label="Virgin (mined)",   color="#d62728")
ax.bar(x2 + width2/2,
       carbon["carbon_intensity_recycled"],
       width2, label="Recycled",          color="#2ca02c")
ax.set_title("Carbon Intensity: Mining vs Recycling (kg CO2 per kg material)")
ax.set_xlabel("Commodity")
ax.set_ylabel("Carbon Intensity")
ax.set_xticks(x2)
ax.set_xticklabels(carbon["commodity"], rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/esg_carbon_comparison.png", dpi=130)
plt.close()
print("    Saved: esg_carbon_comparison.png")

# Chart 4: ESG trend over time for top 4 commodities
fig, ax = plt.subplots(figsize=(12, 6))
top4 = results.head(4)["commodity"].tolist()
colors4 = ["#2ca02c","#1f77b4","#ff7f0e","#9467bd"]
for i, commodity in enumerate(top4):
    mask  = df["commodity"] == commodity
    group = df[mask].sort_values("date")
    trend = group["esg_proxy"].rolling(63).mean()
    ax.plot(group["date"], trend,
            label=commodity,
            color=colors4[i], linewidth=1.5)
ax.set_title("ESG Proxy Trend Over Time (63-day rolling average)")
ax.set_xlabel("Year")
ax.set_ylabel("ESG Proxy Score")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/esg_trend.png", dpi=130)
plt.close()
print("    Saved: esg_trend.png")

print("\n" + "=" * 60)
print("  ESG model complete!")
print(f"  Top ESG commodity:    {results.iloc[0]['commodity']} "
      f"({results.iloc[0]['ESG_score']:.1f})")
print(f"  Lowest ESG commodity: {results.iloc[-1]['commodity']} "
      f"({results.iloc[-1]['ESG_score']:.1f})")
print("=" * 60)