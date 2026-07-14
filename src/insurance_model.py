# src/insurance_model.py
# Insurance Catastrophe Loss Model
# Prices insurance premiums for infrastructure assets
# Run with: python src/insurance_model.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/figures", exist_ok=True)

print("=" * 60)
print("  MatRisk AI - Insurance Catastrophe Model")
print("=" * 60)

# ── HOW INSURANCE PRICING WORKS ───────────────────────────
# An insurer charges a PREMIUM every year.
# When a failure happens, they pay a CLAIM.
# To make money, premiums must cover:
#   1. Expected losses (average claims per year)
#   2. Catastrophe load (extra buffer for bad years)
#   3. Operating costs (usually 20-30% on top)
#
# The formula is:
# Premium = Expected Loss + Catastrophe Load + Expenses
#
# We use Monte Carlo simulation to estimate catastrophe load.
# Monte Carlo = run 10,000 fake years and see what happens.

np.random.seed(42)

# ── STEP 1: Load data ─────────────────────────────────────
print("\n[1] Loading failure data...")
df6 = pd.read_csv("data/raw/DS6_historical_failures_2000.csv")
df3 = pd.read_csv("data/raw/DS3_infrastructure_bridges_5000.csv")
print(f"    Failure events: {len(df6)}")
print(f"    Bridges: {len(df3)}")

# ── STEP 2: Analyze historical losses ────────────────────
print("\n[2] Analyzing historical losses...")
print(f"\n    Loss statistics:")
print(f"    Avg repair cost:      ${df6['repair_cost_USD'].mean():>12,.0f}")
print(f"    Avg replacement cost: ${df6['replacement_value_USD'].mean():>12,.0f}")
print(f"    Avg loss ratio:       {df6['loss_ratio'].mean():.2%}")
print(f"    Max loss ratio:       {df6['loss_ratio'].max():.2%}")

print(f"\n    Losses by severity:")
severity_stats = df6.groupby("severity").agg(
    count=("event_id","count"),
    avg_loss=("repair_cost_USD","mean"),
    avg_ratio=("loss_ratio","mean")
).round(2)
for sev, row in severity_stats.iterrows():
    print(f"    {sev:<12} count={row['count']:>4}  "
          f"avg_loss=${row['avg_loss']:>10,.0f}  "
          f"loss_ratio={row['avg_ratio']:.2%}")

print(f"\n    Failure modes:")
mode_counts = df6["failure_mode"].value_counts()
for mode, count in mode_counts.items():
    bar = "█" * int(count / 5)
    print(f"    {mode:<20} {bar} {count}")

# ── STEP 3: Frequency model ───────────────────────────────
# How many failures happen per year on average?
print("\n[3] Building frequency model...")
years = df6["event_year"].max() - df6["event_year"].min() + 1
total_events = len(df6)
annual_frequency = total_events / years

print(f"    Years of data: {years}")
print(f"    Total events: {total_events}")
print(f"    Annual frequency: {annual_frequency:.1f} events/year")

# Frequency by severity
freq_by_severity = df6.groupby("severity").size() / years
print(f"\n    Annual frequency by severity:")
for sev, freq in freq_by_severity.items():
    print(f"    {sev:<12} {freq:.1f} events/year")

# ── STEP 4: Severity model ────────────────────────────────
# How much does each failure cost?
print("\n[4] Building severity model...")

# Fit a lognormal distribution to repair costs
# Lognormal is standard in insurance for loss modeling
log_losses = np.log(df6["repair_cost_USD"].clip(lower=1))
mu    = log_losses.mean()
sigma = log_losses.std()

print(f"    Lognormal fit: mu={mu:.2f}, sigma={sigma:.2f}")
print(f"    This means:")
print(f"    Median expected loss: ${np.exp(mu):>12,.0f}")
print(f"    Mean expected loss:   "
      f"${np.exp(mu + sigma**2/2):>12,.0f}")

# ── STEP 5: Monte Carlo Simulation ───────────────────────
print("\n[5] Running Monte Carlo simulation (10,000 years)...")
N_SIMULATIONS = 10_000
annual_losses = []

for _ in range(N_SIMULATIONS):
    # How many failures this year?
    # Poisson distribution = standard for rare events
    n_events = np.random.poisson(annual_frequency)

    if n_events == 0:
        annual_losses.append(0)
        continue

    # How much does each failure cost?
    # Sample from lognormal distribution
    losses = np.random.lognormal(mean=mu, sigma=sigma,
                                  size=n_events)
    annual_losses.append(losses.sum())

annual_losses = np.array(annual_losses)

# Key statistics
mean_loss    = annual_losses.mean()
median_loss  = np.median(annual_losses)
var_95       = np.percentile(annual_losses, 95)
var_99       = np.percentile(annual_losses, 99)
tvar_99      = annual_losses[annual_losses >= var_99].mean()
max_loss     = annual_losses.max()
zero_loss_pct = (annual_losses == 0).mean()

print(f"\n    Simulation Results (10,000 simulated years):")
print(f"    Years with no losses:  {zero_loss_pct:.1%}")
print(f"    Median annual loss:    ${median_loss:>12,.0f}")
print(f"    Mean annual loss:      ${mean_loss:>12,.0f}")
print(f"    95th percentile (VaR): ${var_95:>12,.0f}")
print(f"    99th percentile (VaR): ${var_99:>12,.0f}")
print(f"    99th pct TVaR:         ${tvar_99:>12,.0f}")
print(f"    Worst simulated year:  ${max_loss:>12,.0f}")

# ── STEP 6: Premium calculation ───────────────────────────
print("\n[6] Calculating insurance premiums...")

# Standard actuarial pricing formula:
# Pure Premium = Expected Loss
# Loaded Premium = Pure Premium × (1 + catastrophe load) / (1 - expense ratio)
expense_ratio = 0.28     # 28% for operating costs
cat_load      = 0.25     # 25% extra buffer for bad years

pure_premium   = mean_loss
loaded_premium = pure_premium * (1 + cat_load) / (1 - expense_ratio)

print(f"\n    Pure premium (expected loss):   ${pure_premium:>12,.0f}/yr")
print(f"    Catastrophe load (25%):         "
      f"${pure_premium * cat_load:>12,.0f}/yr")
print(f"    Expense load (28%):             "
      f"${loaded_premium - pure_premium*(1+cat_load):>12,.0f}/yr")
print(f"    Final loaded premium:           ${loaded_premium:>12,.0f}/yr")

# Per-bridge premium (spread across 5000 bridges)
per_bridge_premium = loaded_premium / len(df3)
print(f"\n    Per bridge annual premium:      ${per_bridge_premium:>12,.0f}/yr")

# Premium by severity category
print(f"\n    Premium breakdown by failure severity:")
for sev in ["Minor","Moderate","Severe","Critical"]:
    mask = df6["severity"] == sev
    if mask.sum() == 0:
        continue
    sev_freq = mask.sum() / years
    sev_mu   = np.log(df6.loc[mask,"repair_cost_USD"].clip(1)).mean()
    sev_sig  = np.log(df6.loc[mask,"repair_cost_USD"].clip(1)).std()
    sev_el   = sev_freq * np.exp(sev_mu + sev_sig**2/2)
    print(f"    {sev:<12} freq={sev_freq:.1f}/yr  "
          f"EL=${sev_el:>10,.0f}/yr")

# ── STEP 7: Was it predicted? ─────────────────────────────
print("\n[7] Prediction value analysis...")
predicted   = df6[df6["was_predicted"] == 1]
unpredicted = df6[df6["was_predicted"] == 0]

pred_avg_loss   = predicted["repair_cost_USD"].mean()
unpred_avg_loss = unpredicted["repair_cost_USD"].mean()
savings_pct = (unpred_avg_loss - pred_avg_loss) / unpred_avg_loss

print(f"    Predicted failures:   {len(predicted)} events")
print(f"    Unpredicted failures: {len(unpredicted)} events")
print(f"    Avg loss (predicted):   ${pred_avg_loss:>10,.0f}")
print(f"    Avg loss (unpredicted): ${unpred_avg_loss:>10,.0f}")
print(f"    Loss reduction from prediction: {savings_pct:.1%}")
print(f"    => MatRisk AI early warning saves "
      f"~{savings_pct:.0%} on repair costs!")

# ── STEP 8: Save charts ───────────────────────────────────
print("\n[8] Saving charts...")

# Chart 1: Loss distribution from simulation
fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(annual_losses / 1e6, bins=80,
        color="#4C72B0", edgecolor="white", alpha=0.8)
ax.axvline(mean_loss/1e6, color="#2ca02c", linewidth=2,
           linestyle="-", label=f"Mean: ${mean_loss/1e6:.1f}M")
ax.axvline(var_95/1e6, color="#ff7f0e", linewidth=2,
           linestyle="--", label=f"95th VaR: ${var_95/1e6:.1f}M")
ax.axvline(var_99/1e6, color="#d62728", linewidth=2,
           linestyle="--", label=f"99th VaR: ${var_99/1e6:.1f}M")
ax.set_title("Monte Carlo: Annual Loss Distribution (10,000 simulated years)")
ax.set_xlabel("Annual Loss ($M)")
ax.set_ylabel("Number of Simulated Years")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/insurance_loss_distribution.png", dpi=130)
plt.close()
print("    Saved: insurance_loss_distribution.png")

# Chart 2: Loss by failure mode
fig, ax = plt.subplots(figsize=(10, 5))
mode_loss = df6.groupby("failure_mode")["repair_cost_USD"].mean() / 1e6
mode_loss = mode_loss.sort_values(ascending=False)
ax.bar(mode_loss.index, mode_loss.values,
       color="#C44E52", edgecolor="white")
ax.set_title("Average Repair Cost by Failure Mode ($M)")
ax.set_xlabel("Failure Mode")
ax.set_ylabel("Average Repair Cost ($M)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("notebooks/figures/insurance_loss_by_mode.png", dpi=130)
plt.close()
print("    Saved: insurance_loss_by_mode.png")

# Chart 3: Predicted vs unpredicted losses
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["Predicted\n(early warning)", "NOT Predicted\n(surprise failure)"]
values = [pred_avg_loss/1e6, unpred_avg_loss/1e6]
colors = ["#2ca02c", "#d62728"]
bars   = ax.bar(labels, values, color=colors, edgecolor="white", width=0.4)
ax.set_title("Average Repair Cost: Predicted vs Unpredicted Failures")
ax.set_ylabel("Average Repair Cost ($M)")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"${val:.2f}M", ha="center",
            fontsize=12, fontweight="bold")
savings_arrow_y = values[1] * 0.5
ax.annotate(f"Early warning\nsaves {savings_pct:.0%}",
            xy=(0.5, savings_arrow_y),
            fontsize=11, ha="center", color="#2ca02c",
            fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/figures/insurance_prediction_value.png", dpi=130)
plt.close()
print("    Saved: insurance_prediction_value.png")

# Chart 4: Premium breakdown pie chart
fig, ax = plt.subplots(figsize=(8, 8))
exp_load = loaded_premium - pure_premium*(1+cat_load)
sizes    = [pure_premium, pure_premium*cat_load, exp_load]
labels_p = [f"Expected Loss\n${pure_premium/1e6:.1f}M",
            f"Cat Buffer (25%)\n${pure_premium*cat_load/1e6:.1f}M",
            f"Expenses (28%)\n${exp_load/1e6:.1f}M"]
colors_p = ["#4C72B0", "#ff7f0e", "#55A868"]
ax.pie(sizes, labels=labels_p, colors=colors_p,
       autopct="%1.1f%%", startangle=90,
       textprops={"fontsize":11})
ax.set_title(f"Insurance Premium Breakdown\n"
             f"Total: ${loaded_premium/1e6:.1f}M/year", fontsize=13)
plt.tight_layout()
plt.savefig("notebooks/figures/insurance_premium_breakdown.png", dpi=130)
plt.close()
print("    Saved: insurance_premium_breakdown.png")

print("\n" + "=" * 60)
print("  Insurance model complete!")
print(f"  Annual premium: ${loaded_premium:,.0f}")
print(f"  99th VaR:       ${var_99:,.0f}")
print(f"  Prediction saves: {savings_pct:.0%} on repair costs")
print("=" * 60)