import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

os.makedirs("notebooks/figures", exist_ok=True)

print("Loading datasets...")
df1 = pd.read_csv("data/raw/DS1_material_properties_5500.csv")
df2 = pd.read_csv("data/raw/DS2_commodity_prices_10yr.csv")
df3 = pd.read_csv("data/raw/DS3_infrastructure_bridges_5000.csv")
df6 = pd.read_csv("data/raw/DS6_historical_failures_2000.csv")
print("All loaded. Creating charts...")

fig, ax = plt.subplots(figsize=(9, 5))
counts = df1["crystal_system"].value_counts()
bars = ax.bar(counts.index, counts.values, color="#4C72B0", edgecolor="white")
ax.set_title("Materials per Crystal System (DS1)", fontsize=14)
ax.set_xlabel("Crystal System")
ax.set_ylabel("Number of Materials")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(int(bar.get_height())), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("notebooks/figures/chart1_crystal_systems.png", dpi=130)
plt.close()
print("Chart 1 done")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df1["poisson_ratio"], bins=50, color="#55A868", edgecolor="white")
ax.axvline(-1, color="red", linestyle="--", linewidth=2, label="Lower limit (-1)")
ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Upper limit (0.5)")
ax.set_title("Poisson Ratio Distribution (DS1)", fontsize=13)
ax.set_xlabel("Poisson Ratio")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/chart2_poisson_ratio.png", dpi=130)
plt.close()
print("Chart 2 done")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df1["formation_energy_per_atom_eV"], bins=50, color="#C44E52", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Stability boundary")
ax.set_title("Formation Energy - negative means stable (DS1)", fontsize=13)
ax.set_xlabel("Formation Energy per Atom (eV)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/chart3_formation_energy.png", dpi=130)
plt.close()
print("Chart 3 done")

fig, ax = plt.subplots(figsize=(9, 6))
metals = df1[df1["is_metal"] == 1]
nonmetals = df1[df1["is_metal"] == 0]
ax.scatter(nonmetals["bulk_modulus_GPa"], nonmetals["shear_modulus_GPa"],
           s=8, alpha=0.4, color="#FF7F0E", label="Non-metal")
ax.scatter(metals["bulk_modulus_GPa"], metals["shear_modulus_GPa"],
           s=8, alpha=0.4, color="#1F77B4", label="Metal")
ax.set_title("Bulk vs Shear Modulus (DS1)", fontsize=13)
ax.set_xlabel("Bulk Modulus (GPa)")
ax.set_ylabel("Shear Modulus (GPa)")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig("notebooks/figures/chart4_bulk_vs_shear.png", dpi=130)
plt.close()
print("Chart 4 done")

df2["date"] = pd.to_datetime(df2["date"])
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]
for i, (name, group) in enumerate(df2.groupby("commodity")):
    g = group.sort_values("date")
    norm = g["close"] / g["close"].iloc[0] * 100
    ax.plot(g["date"], norm, label=name, color=colors[i % len(colors)], linewidth=1.2)
ax.set_title("Commodity Prices 2014-2024 (rebased to 100)", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Price Index")
ax.legend(fontsize=8, ncol=2)
ax.axhline(100, color="black", linestyle=":", linewidth=0.8)
plt.tight_layout()
plt.savefig("notebooks/figures/chart5_commodity_prices.png", dpi=130)
plt.close()
print("Chart 5 done")

fig, ax = plt.subplots(figsize=(9, 5))
counts = df3["condition_rating"].value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values, color="#4C72B0", edgecolor="white")
ax.set_title("Bridge Condition Ratings 9=Excellent 1=Near Failure (DS3)", fontsize=12)
ax.set_xlabel("Condition Rating")
ax.set_ylabel("Number of Bridges")
plt.tight_layout()
plt.savefig("notebooks/figures/chart6_bridge_conditions.png", dpi=130)
plt.close()
print("Chart 6 done")

fig, ax = plt.subplots(figsize=(9, 6))
d = df3[df3["structurally_deficient"] == 1]
ok = df3[df3["structurally_deficient"] == 0]
ax.scatter(ok["age_years"], ok["corrosion_rate_mm_yr"], s=6, alpha=0.25, color="#2ca02c", label="OK")
ax.scatter(d["age_years"], d["corrosion_rate_mm_yr"], s=15, alpha=0.7, color="#d62728", label="Deficient")
ax.set_title("Bridge Age vs Corrosion Rate (DS3)", fontsize=13)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Corrosion Rate (mm/year)")
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig("notebooks/figures/chart7_age_vs_corrosion.png", dpi=130)
plt.close()
print("Chart 7 done")

fig, ax = plt.subplots(figsize=(9, 5))
fc = df6["failure_mode"].value_counts()
ax.barh(fc.index, fc.values, color="#8172B2", edgecolor="white")
ax.set_title("Infrastructure Failure Modes (DS6)", fontsize=13)
ax.set_xlabel("Number of Failures")
plt.tight_layout()
plt.savefig("notebooks/figures/chart8_failure_modes.png", dpi=130)
plt.close()
print("Chart 8 done")

fig, ax = plt.subplots(figsize=(9, 5))
pr = df6.groupby("severity")["was_predicted"].mean() * 100
ax.bar(pr.index, pr.values, color="#55A868", edgecolor="white")
ax.set_title("Percentage of Failures Predicted in Advance by Severity (DS6)", fontsize=12)
ax.set_xlabel("Severity")
ax.set_ylabel("Percent Predicted")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("notebooks/figures/chart9_prediction_rate.png", dpi=130)
plt.close()
print("Chart 9 done")

print("\nAll 9 charts saved to notebooks/figures/")