# src/commodity_model.py
# Commodity Price Prediction Model
# Predicts if commodity prices will rise or fall in next 21 days
# Run with: python src/commodity_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/figures", exist_ok=True)

print("=" * 60)
print("  MatRisk AI - Commodity Price Prediction Model")
print("=" * 60)

# ── STEP 1: Load both datasets ────────────────────────────
print("\n[1] Loading data...")
df2 = pd.read_csv("data/raw/DS2_commodity_prices_10yr.csv",
                   parse_dates=["date"])
df4 = pd.read_csv("data/raw/DS4_crossdomain_features_daily.csv",
                   parse_dates=["date"])
print(f"    DS2 (prices): {len(df2)} rows")
print(f"    DS4 (features): {len(df4)} rows")

# ── STEP 2: Merge the two datasets ───────────────────────
print("\n[2] Merging price data with material features...")
df = pd.merge(df2, df4, on=["date", "commodity"], how="inner")
print(f"    Merged dataset: {len(df)} rows")
print(f"    Commodities: {df['commodity'].unique().tolist()}")

# ── STEP 3: Create the target variable ───────────────────
print("\n[3] Creating prediction target...")
df = df.sort_values(["commodity", "date"]).reset_index(drop=True)
df["future_return"] = df.groupby("commodity")["close"].transform(
    lambda x: x.shift(-21) / x - 1
)
df["target"] = (df["future_return"] > 0).astype(int)
df = df.dropna(subset=["future_return"])
print(f"    Rows after removing future NaN: {len(df)}")
print(f"    Price goes UP:   {df['target'].sum()} days ({df['target'].mean():.1%})")
print(f"    Price goes DOWN: {(df['target']==0).sum()} days ({(df['target']==0).mean():.1%})")

# ── STEP 4: Choose features ───────────────────────────────
print("\n[4] Preparing features...")
le = LabelEncoder()
df["commodity_code"] = le.fit_transform(df["commodity"])

feature_cols = [
    "commodity_code",
    "daily_return",
    "return_5d",
    "return_21d",
    "volatility_21d_ann",
    "rsi_14",
    "macd",
    "bollinger_z",
    "momentum_10d",
    "sma_21",
    "mqi",
    "mqi_21d_trend",
    "supply_disruption_prob",
    "substitution_elasticity",
    "herfindahl_index",
    "green_premium_per_kg",
]

df_clean = df[feature_cols + ["target", "commodity", "date"]].dropna()
print(f"    Clean rows: {len(df_clean)}")
print(f"    Features: {len(feature_cols)}")

# ── STEP 5: Split by time ─────────────────────────────────
print("\n[5] Splitting by time (no data leakage)...")
split_date = "2022-01-01"
train = df_clean[df_clean["date"] < split_date]
test  = df_clean[df_clean["date"] >= split_date]

X_train = train[feature_cols]
y_train = train["target"]
X_test  = test[feature_cols]
y_test  = test["target"]

print(f"    Train: {len(X_train)} rows (before {split_date})")
print(f"    Test:  {len(X_test)} rows (after {split_date})")

# ── STEP 6: Train model ───────────────────────────────────
print("\n[6] Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
model.fit(X_train, y_train)
print("    Done!")

# ── STEP 7: Evaluate ──────────────────────────────────────
print("\n[7] Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n    Overall Accuracy: {accuracy:.2%}")
print(f"\n    Detailed Report:")
print(classification_report(y_test, y_pred,
      target_names=["Price Down", "Price Up"]))

# ── STEP 8: Results per commodity ────────────────────────
print("\n[8] Accuracy per commodity:")
test_results = test[["commodity", "date"]].copy()
test_results["actual"]    = y_test.values
test_results["predicted"] = y_pred
test_results["correct"]   = (
    test_results["actual"] == test_results["predicted"]
)
per_commodity = (test_results.groupby("commodity")["correct"]
                 .mean().sort_values(ascending=False))
print()
for commodity, acc in per_commodity.items():
    bar = "█" * int(acc * 30)
    print(f"    {commodity:<20} {bar} {acc:.1%}")

# ── STEP 9: Feature importance ────────────────────────────
print("\n[9] Most important features:")
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)
print()
for feat, imp in importance.head(8).items():
    bar = "█" * int(imp * 100)
    print(f"    {feat:<28} {bar} {imp:.3f}")

# ── STEP 10: Save charts ──────────────────────────────────
print("\n[10] Saving charts...")

# Chart 1: Accuracy per commodity (horizontal bars)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2ca02c" if v >= 0.55 else "#ff7f0e" if v >= 0.50
          else "#d62728" for v in per_commodity.values]
bars = ax.barh(per_commodity.index, per_commodity.values,
               color=colors, edgecolor="white")
ax.axvline(0.5, color="black", linestyle="--",
           linewidth=2, label="Random baseline (50%)")
ax.set_title("Commodity Price Prediction Accuracy by Commodity")
ax.set_xlabel("Accuracy")
ax.set_xlim(0, 1)
for bar, val in zip(bars, per_commodity.values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/commodity_accuracy.png", dpi=130)
plt.close()
print("    Saved: commodity_accuracy.png")

# Chart 2: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
importance.head(10).plot(kind="barh", ax=ax,
                          color="#4C72B0", edgecolor="white")
ax.set_title("Top 10 Features for Commodity Price Prediction")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("notebooks/figures/commodity_features.png", dpi=130)
plt.close()
print("    Saved: commodity_features.png")

# Chart 3: Accuracy per commodity (vertical bars)
fig, ax = plt.subplots(figsize=(10, 5))
commodities = per_commodity.index.tolist()
accuracies  = per_commodity.values.tolist()
colors2 = ["#2ca02c" if v >= 0.55 else "#ff7f0e" if v >= 0.50
           else "#d62728" for v in accuracies]
ax.bar(commodities, accuracies, color=colors2, edgecolor="white")
ax.axhline(0.5, color="black", linestyle="--",
           linewidth=2, label="Random baseline (50%)")
ax.set_title("Prediction Accuracy per Commodity (2022-2024)")
ax.set_xlabel("Commodity")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=45)
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/commodity_signal.png", dpi=130)
plt.close()
print("    Saved: commodity_signal.png")

# ── DONE ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Commodity model complete!")
print(f"  Overall accuracy: {accuracy:.2%}")
print(f"  Best commodity:   {per_commodity.index[0]} "
      f"({per_commodity.iloc[0]:.1%})")
print("=" * 60)