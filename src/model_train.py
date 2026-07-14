# src/model_train.py
# First ML model - predicts bulk modulus of materials
# Run with: python src/model_train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/figures", exist_ok=True)

print("=" * 55)
print("  MatRisk AI - Material Property Prediction Model")
print("=" * 55)

# ── STEP 1: Load data ────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv("data/raw/DS1_material_properties_5500.csv")
print(f"    Loaded {len(df)} materials")

# ── STEP 2: Prepare features ─────────────────────────────
# We need to convert text columns into numbers
# Machine learning models only understand numbers
print("\n[2] Preparing features...")

# Convert crystal_system text to numbers (e.g. cubic=0, hexagonal=1 etc.)
le_crystal = LabelEncoder()
df["crystal_system_code"] = le_crystal.fit_transform(df["crystal_system"])

le_category = LabelEncoder()
df["category_code"] = le_category.fit_transform(df["category"])

# Choose which columns to use as inputs (features)
feature_cols = [
    "n_elements",           # how many elements in the material
    "crystal_system_code",  # what crystal structure (converted to number)
    "category_code",        # what category of material
    "formation_energy_per_atom_eV",  # how stable it is
    "energy_above_hull_eV", # another stability measure
    "band_gap_eV",          # electrical property
    "is_metal",             # is it a metal? (1=yes, 0=no)
    "shear_modulus_GPa",    # related mechanical property
    "poisson_ratio",        # how it deforms sideways
    "density_g_cm3",        # how heavy it is
    "nsites",               # number of atoms in unit cell
    "melting_point_K",      # temperature it melts at
]

# Target: what we want to predict
target_col = "bulk_modulus_GPa"

# Remove rows with missing values
df_clean = df[feature_cols + [target_col]].dropna()
print(f"    Using {len(df_clean)} materials after removing missing values")
print(f"    Input features: {len(feature_cols)}")
print(f"    Predicting: {target_col}")

# ── STEP 3: Split into training and test sets ─────────────
# We train on 80% of the data, test on the remaining 20%
# The model never sees the test data during training
print("\n[3] Splitting data...")
X = df_clean[feature_cols]
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Training set: {len(X_train)} materials")
print(f"    Test set:     {len(X_test)} materials")

# ── STEP 4: Train the model ───────────────────────────────
print("\n[4] Training Random Forest model...")
print("    (this takes about 10-20 seconds...)")

model = RandomForestRegressor(
    n_estimators=200,   # 200 decision trees
    max_depth=15,       # how deep each tree can go
    random_state=42,    # for reproducibility
    n_jobs=-1           # use all CPU cores
)
model.fit(X_train, y_train)
print("    Training complete!")

# ── STEP 5: Evaluate the model ───────────────────────────
print("\n[5] Evaluating model...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n    Results:")
print(f"    R² Score:  {r2:.4f}  (1.0 = perfect, 0.0 = useless)")
print(f"    MAE:       {mae:.2f} GPa (average prediction error)")

if r2 > 0.85:
    print("    Excellent model!")
elif r2 > 0.70:
    print("    Good model!")
elif r2 > 0.50:
    print("    Decent model - room for improvement")
else:
    print("    Needs improvement")

# ── STEP 6: Feature importance ────────────────────────────
print("\n[6] Which features matter most?")
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print()
for feat, imp in importance.items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:<35} {bar} {imp:.3f}")

# ── STEP 7: Save charts ───────────────────────────────────
print("\n[7] Saving charts...")

# Chart A: Predicted vs Actual
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#4C72B0")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val],
        "r--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Bulk Modulus (GPa)")
ax.set_ylabel("Predicted Bulk Modulus (GPa)")
ax.set_title(f"Predicted vs Actual Bulk Modulus\nR² = {r2:.4f} | MAE = {mae:.2f} GPa")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/model_predicted_vs_actual.png", dpi=130)
plt.close()
print("    Saved: model_predicted_vs_actual.png")

# Chart B: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
importance.plot(kind="barh", ax=ax, color="#55A868", edgecolor="white")
ax.set_title("Feature Importance - What helps predict Bulk Modulus?")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("notebooks/figures/model_feature_importance.png", dpi=130)
plt.close()
print("    Saved: model_feature_importance.png")

# Chart C: Error distribution
errors = y_pred - y_test
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=50, color="#C44E52", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Zero error")
ax.set_title("Prediction Error Distribution")
ax.set_xlabel("Prediction Error (GPa)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/figures/model_error_distribution.png", dpi=130)
plt.close()
print("    Saved: model_error_distribution.png")

# ── STEP 8: Test on a single material ────────────────────
print("\n[8] Testing on a single real material...")
sample = X_test.iloc[0:1]
actual = y_test.iloc[0]
predicted = model.predict(sample)[0]

print(f"\n    Material features:")
for col in feature_cols:
    print(f"      {col}: {sample[col].values[0]}")
print(f"\n    Actual bulk modulus:    {actual:.2f} GPa")
print(f"    Predicted bulk modulus: {predicted:.2f} GPa")
print(f"    Error: {abs(actual - predicted):.2f} GPa")

print("\n" + "=" * 55)
print("  Model training complete!")
print("  3 charts saved to notebooks/figures/")
print("=" * 55)