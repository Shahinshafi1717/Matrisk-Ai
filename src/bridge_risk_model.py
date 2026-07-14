# src/bridge_risk_model.py
# Bridge Failure & Credit Risk Model
# Run with: python src/bridge_risk_model.py

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/figures", exist_ok=True)

print("=" * 60)
print("  MatRisk AI - Bridge Failure & Credit Risk Model")
print("=" * 60)

# ── STEP 1: Load data ─────────────────────────────────────
print("\n[1] Loading bridge data...")
df = pd.read_csv("data/raw/DS3_infrastructure_bridges_5000.csv",
                 parse_dates=["last_inspection_date"])
print(f"    Loaded {len(df)} bridges")

# ── STEP 2: Prepare survival variables ───────────────────
print("\n[2] Preparing survival variables...")
df["duration"]      = df["age_years"]
df["event"]         = df["structurally_deficient"]
df["remaining_life"] = df["design_life_years"] - df["age_years"]
df["pct_life_used"] = df["age_years"] / df["design_life_years"]

print(f"    Deficient bridges: {df['event'].sum()} ({df['event'].mean():.1%})")
print(f"    Average age: {df['age_years'].mean():.1f} years")
print(f"    Average remaining life: {df['remaining_life'].mean():.1f} years")

# ── STEP 3: Kaplan-Meier Survival Curve ──────────────────
print("\n[3] Kaplan-Meier survival curve...")
kmf = KaplanMeierFitter()
kmf.fit(durations=df["duration"],
        event_observed=df["event"],
        label="All Bridges")

print(f"\n    Survival probabilities:")
for age in [20, 40, 60, 80, 100]:
    prob = kmf.survival_function_at_times([age]).values[0]
    bar  = "█" * int(prob * 30)
    print(f"    Age {age:3d} years: {bar} {prob:.1%}")

# ── STEP 4: Survival by material ──────────────────────────
print("\n[4] Survival by material (at 50 years)...")
top_materials = df["material"].value_counts().head(5).index.tolist()
for material in top_materials:
    mask  = df["material"] == material
    kmf_m = KaplanMeierFitter()
    kmf_m.fit(df.loc[mask,"duration"],
              df.loc[mask,"event"],
              label=material)
    prob = kmf_m.survival_function_at_times([50]).values[0]
    print(f"    {material:<35} {prob:.1%}")

# ── STEP 5: Cox Model ─────────────────────────────────────
print("\n[5] Cox Proportional Hazards model...")
le_mat  = LabelEncoder()
le_loc  = LabelEncoder()
le_cor  = LabelEncoder()
le_btype = LabelEncoder()
df["material_code"]    = le_mat.fit_transform(df["material"])
df["location_code"]    = le_loc.fit_transform(df["location"])
df["corrosion_code"]   = le_cor.fit_transform(df["corrosion_environment"])
df["bridge_type_code"] = le_btype.fit_transform(df["bridge_type"])

cox_features = [
    "duration", "event",
    "material_code", "corrosion_rate_mm_yr",
    "condition_rating", "age_years", "pct_life_used",
    "corrosion_code", "bridge_type_code",
    "fatigue_cycles_millions", "remaining_thickness_mm",
]
df_cox = df[cox_features].dropna()
print(f"    Using {len(df_cox)} bridges")

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_cox, duration_col="duration",
        event_col="event", show_progress=False)

print("\n    Risk factors:")
summary = cph.summary[["coef","exp(coef)","p"]].sort_values(
    "coef", ascending=False)
for feat, row in summary.iterrows():
    direction = "↑ MORE RISK" if row["coef"] > 0 else "↓ LESS RISK"
    sig = "***" if row["p"]<0.01 else "**" if row["p"]<0.05 else ""
    print(f"    {feat:<30} {row['coef']:+.3f}  {direction} {sig}")

# ── STEP 6: Realistic Credit Risk ────────────────────────
# Since only 3.2% of bridges are deficient, the Cox model
# gives very low failure probabilities. Instead we build a
# realistic risk score from multiple condition indicators.
print("\n[6] Calculating realistic credit risk scores...")

def calculate_risk_score(row):
    """
    Build a 0-100 risk score from multiple factors.
    Higher score = higher risk of failure.
    """
    score = 0

    # Condition rating (NBI scale 0-9, lower = worse)
    # Weight: 40% of total score
    condition_risk = (9 - row["condition_rating"]) / 9
    score += condition_risk * 40

    # % of design life used
    # Weight: 25% of total score
    life_risk = min(row["pct_life_used"], 1.0)
    score += life_risk * 25

    # Corrosion rate (normalized)
    # Weight: 20% of total score
    max_corrosion = 0.15  # mm/yr
    corr_risk = min(row["corrosion_rate_mm_yr"] / max_corrosion, 1.0)
    score += corr_risk * 20

    # Remaining thickness ratio
    # Weight: 15% of total score
    thickness_ratio = row["remaining_thickness_mm"] / row["original_thickness_mm"]
    thickness_risk  = 1 - min(thickness_ratio, 1.0)
    score += thickness_risk * 15

    return round(score, 2)

df["risk_score"] = df.apply(calculate_risk_score, axis=1)

# Convert risk score to failure probability (0-1)
df["failure_prob_10yr"] = df["risk_score"] / 100

# Credit rating based on risk score
def credit_rating(score):
    if score < 15:
        return "AAA - Very Safe"
    elif score < 25:
        return "BBB - Safe"
    elif score < 40:
        return "BB - Moderate Risk"
    elif score < 55:
        return "B - High Risk"
    else:
        return "CCC - Critical"

df["credit_rating"] = df["risk_score"].apply(credit_rating)

# Risk premium in basis points (1 bps = 0.01% interest)
df["risk_premium_bps"] = (df["failure_prob_10yr"] * 500).round(0)

print("\n    Credit rating distribution:")
rating_order = ["AAA - Very Safe","BBB - Safe",
                "BB - Moderate Risk","B - High Risk","CCC - Critical"]
rating_counts = df["credit_rating"].value_counts().reindex(
    rating_order, fill_value=0)
for rating, count in rating_counts.items():
    bar = "█" * int(count / 20)
    print(f"    {rating:<25} {bar} {count}")

print(f"\n    Risk score stats:")
print(f"    Min:  {df['risk_score'].min():.1f}")
print(f"    Max:  {df['risk_score'].max():.1f}")
print(f"    Mean: {df['risk_score'].mean():.1f}")
print(f"\n    Average failure probability: {df['failure_prob_10yr'].mean():.2%}")
print(f"    Average risk premium: {df['risk_premium_bps'].mean():.0f} bps")

# ── STEP 7: Expected Loss ─────────────────────────────────
print("\n[7] Expected Loss calculation...")
# Expected Loss = PD × LGD × EAD
# PD  = Probability of Default (failure_prob_10yr)
# LGD = Loss Given Default (45% standard assumption)
# EAD = Exposure at Default (loan outstanding)

df_el = df[["bridge_id","failure_prob_10yr",
            "loan_outstanding_M","replacement_cost_M",
            "credit_rating","risk_premium_bps",
            "risk_score"]].dropna()

df_el["PD"]   = df_el["failure_prob_10yr"]
df_el["LGD"]  = 0.45
df_el["EAD"]  = df_el["loan_outstanding_M"]
df_el["EL_M"] = df_el["PD"] * df_el["LGD"] * df_el["EAD"]

total_portfolio = df_el["EAD"].sum()
total_el        = df_el["EL_M"].sum()
el_rate         = total_el / total_portfolio

print(f"\n    Portfolio Summary:")
print(f"    Bridges in portfolio:    {len(df_el)}")
print(f"    Total loans:             ${total_portfolio:.1f}M")
print(f"    Total expected loss:     ${total_el:.1f}M")
print(f"    Expected loss rate:      {el_rate:.2%}")

print(f"\n    Top 5 riskiest bridges:")
top5 = df_el.nlargest(5, "EL_M")[
    ["bridge_id","PD","EAD","EL_M","credit_rating"]]
for _, row in top5.iterrows():
    print(f"    {row['bridge_id']}  "
          f"PD={row['PD']:.1%}  "
          f"Loan=${row['EAD']:.1f}M  "
          f"EL=${row['EL_M']:.2f}M  "
          f"{row['credit_rating']}")

# ── STEP 8: Save charts ───────────────────────────────────
print("\n[8] Saving charts...")

# Chart 1: Kaplan-Meier
fig, ax = plt.subplots(figsize=(10, 6))
kmf.plot_survival_function(ax=ax, color="#4C72B0", linewidth=2.5)
for material in top_materials[:4]:
    mask = df["material"] == material
    kmf_t = KaplanMeierFitter()
    kmf_t.fit(df.loc[mask,"duration"],
               df.loc[mask,"event"], label=material)
    kmf_t.plot_survival_function(ax=ax, linewidth=1.5)
ax.set_title("Bridge Survival Curves — Probability of Staying Safe by Age")
ax.set_xlabel("Bridge Age (years)")
ax.set_ylabel("Survival Probability")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="red", linestyle="--",
           linewidth=1.5, label="50% threshold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("notebooks/figures/bridge_survival_curve.png", dpi=130)
plt.close()
print("    Saved: bridge_survival_curve.png")

# Chart 2: Credit rating distribution
fig, ax = plt.subplots(figsize=(10, 5))
colors_map = {
    "AAA - Very Safe":    "#2ca02c",
    "BBB - Safe":         "#55A868",
    "BB - Moderate Risk": "#ff7f0e",
    "B - High Risk":      "#d62728",
    "CCC - Critical":     "#8B0000",
}
bar_colors = [colors_map[r] for r in rating_counts.index]
bars = ax.bar(rating_counts.index, rating_counts.values,
              color=bar_colors, edgecolor="white")
ax.set_title("Bridge Credit Risk Rating Distribution")
ax.set_xlabel("Credit Rating")
ax.set_ylabel("Number of Bridges")
ax.tick_params(axis="x", rotation=20)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 10,
            str(int(bar.get_height())),
            ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("notebooks/figures/bridge_credit_ratings.png", dpi=130)
plt.close()
print("    Saved: bridge_credit_ratings.png")

# Chart 3: Risk score distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["risk_score"], bins=40,
        color="#8172B2", edgecolor="white")
ax.axvline(15, color="#2ca02c", linestyle="--",
           linewidth=1.5, label="AAA threshold (15)")
ax.axvline(25, color="#55A868", linestyle="--",
           linewidth=1.5, label="BBB threshold (25)")
ax.axvline(40, color="#ff7f0e", linestyle="--",
           linewidth=1.5, label="BB threshold (40)")
ax.axvline(55, color="#d62728", linestyle="--",
           linewidth=1.5, label="B threshold (55)")
ax.set_title("Bridge Risk Score Distribution (0=Safe, 100=Critical)")
ax.set_xlabel("Risk Score")
ax.set_ylabel("Number of Bridges")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("notebooks/figures/bridge_risk_scores.png", dpi=130)
plt.close()
print("    Saved: bridge_risk_scores.png")

# Chart 4: Expected loss by credit rating
fig, ax = plt.subplots(figsize=(10, 5))
el_by_rating = df_el.groupby("credit_rating")["EL_M"].sum().reindex(
    rating_order, fill_value=0)
bar_colors2 = [colors_map[r] for r in el_by_rating.index]
bars2 = ax.bar(el_by_rating.index, el_by_rating.values,
               color=bar_colors2, edgecolor="white")
ax.set_title("Total Expected Loss by Credit Rating ($M)")
ax.set_xlabel("Credit Rating")
ax.set_ylabel("Expected Loss ($M)")
ax.tick_params(axis="x", rotation=20)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"${bar.get_height():.1f}M",
            ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("notebooks/figures/bridge_expected_loss.png", dpi=130)
plt.close()
print("    Saved: bridge_expected_loss.png")

print("\n" + "=" * 60)
print("  Bridge risk model complete!")
print(f"  Portfolio expected loss rate: {el_rate:.2%}")
print(f"  Total expected loss: ${total_el:.1f}M")
print("=" * 60)