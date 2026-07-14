# src/physics_model.py
# Physics-Informed Material Property Prediction
# This model knows the laws of physics and enforces them
# Run with: python src/physics_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/figures", exist_ok=True)

print("=" * 60)
print("  MatRisk AI - Physics-Informed Prediction Model")
print("=" * 60)

# ── THE PHYSICS ───────────────────────────────────────────
# These are the real equations from materials science.
# For any isotropic material:
#
#   E  = Young's modulus (overall stiffness)
#   K  = Bulk modulus (resistance to compression)
#   G  = Shear modulus (resistance to twisting)
#   nu = Poisson ratio (sideways deformation)
#
#   Relationships:
#   E  = 2G(1 + nu)        -- equation 1
#   E  = 3K(1 - 2*nu)      -- equation 2
#   nu = (3K - 2G) /       -- equation 3
#        (2*(3K + G))
#
# These MUST all be consistent. Our model will enforce this.

def derive_poisson(K, G):
    """Derive Poisson ratio from Bulk and Shear moduli."""
    return (3*K - 2*G) / (2*(3*K + G))

def derive_youngs(K, G):
    """Derive Young's modulus from Bulk and Shear moduli."""
    return (9*K*G) / (3*K + G)

def physics_violation_score(K, G, nu_reported):
    """
    Calculate how much the reported Poisson ratio
    violates the physics equation.
    Lower is better. Zero means perfect physics.
    """
    nu_derived = derive_poisson(K, G)
    return np.abs(nu_derived - nu_reported)

# ── STEP 1: Load and prepare data ────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv("data/raw/DS1_material_properties_5500.csv")

le_crystal = LabelEncoder()
df["crystal_system_code"] = le_crystal.fit_transform(df["crystal_system"])
le_category = LabelEncoder()
df["category_code"] = le_category.fit_transform(df["category"])

feature_cols = [
    "n_elements",
    "crystal_system_code",
    "category_code",
    "formation_energy_per_atom_eV",
    "energy_above_hull_eV",
    "band_gap_eV",
    "is_metal",
    "shear_modulus_GPa",
    "poisson_ratio",
    "density_g_cm3",
    "nsites",
    "melting_point_K",
]
target_col = "bulk_modulus_GPa"

df_clean = df[feature_cols + [target_col]].dropna()
print(f"    {len(df_clean)} materials ready")

X = df_clean[feature_cols]
y = df_clean[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── STEP 2: Train XGBoost model ──────────────────────────
print("\n[2] Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"    XGBoost R²: {r2_xgb:.4f} | MAE: {mae_xgb:.2f} GPa")

# ── STEP 3: Apply physics correction ─────────────────────
# After the model predicts K (bulk modulus), we check if
# the prediction is physically consistent with G and nu.
# If not, we nudge K toward the physically correct value.
print("\n[3] Applying physics correction...")

K_pred = y_pred_xgb.copy()
G_test = X_test["shear_modulus_GPa"].values
nu_test = X_test["poisson_ratio"].values

corrections = 0
corrected_K = []

for i in range(len(K_pred)):
    K = K_pred[i]
    G = G_test[i]
    nu = nu_test[i]

    # What does physics say K should be given G and nu?
    # From: nu = (3K - 2G) / (2*(3K + G))
    # Solving for K: K = 2G(1+nu) / (3(1-2nu))
    if abs(1 - 2*nu) > 1e-6:  # avoid division by zero
        K_physics = 2*G*(1 + nu) / (3*(1 - 2*nu))

        # How far is our prediction from physics?
        violation = abs(K - K_physics)

        # If violation is large, blend toward physics answer
        if violation > 20:  # GPa threshold
            # 30% physics correction, 70% model prediction
            K_corrected = 0.7 * K + 0.3 * K_physics
            corrections += 1
        else:
            K_corrected = K
    else:
        K_corrected = K

    corrected_K.append(K_corrected)

y_pred_physics = np.array(corrected_K)
mae_physics = mean_absolute_error(y_test, y_pred_physics)
r2_physics = r2_score(y_test, y_pred_physics)

print(f"    Physics corrections applied: {corrections}")
print(f"    Physics-corrected R²: {r2_physics:.4f} | MAE: {mae_physics:.2f} GPa")

# ── STEP 4: Check physics violations in data ──────────────
print("\n[4] Checking physics consistency in dataset...")
df_check = df_clean.copy()
df_check["nu_derived"] = derive_poisson(
    df_check["bulk_modulus_GPa"],
    df_check["shear_modulus_GPa"]
)
df_check["nu_violation"] = physics_violation_score(
    df_check["bulk_modulus_GPa"],
    df_check["shear_modulus_GPa"],
    df_check["poisson_ratio"]
)
df_check["youngs_modulus_GPa"] = derive_youngs(
    df_check["bulk_modulus_GPa"],
    df_check["shear_modulus_GPa"]
)

violations = (df_check["nu_violation"] > 0.05).sum()
print(f"    Materials violating physics (error > 0.05): {violations}")
print(f"    Average violation: {df_check['nu_violation'].mean():.4f}")
print(f"    Max violation: {df_check['nu_violation'].max():.4f}")
print(f"\n    Derived Young's modulus stats:")
print(f"    Min: {df_check['youngs_modulus_GPa'].min():.1f} GPa")
print(f"    Max: {df_check['youngs_modulus_GPa'].max():.1f} GPa")
print(f"    Mean: {df_check['youngs_modulus_GPa'].mean():.1f} GPa")

# ── STEP 5: Compare models ────────────────────────────────
print("\n[5] Model Comparison:")
print(f"    {'Model':<30} {'R²':>8} {'MAE (GPa)':>12}")
print(f"    {'-'*50}")
print(f"    {'XGBoost (no physics)':<30} {r2_xgb:>8.4f} {mae_xgb:>12.2f}")
print(f"    {'XGBoost + Physics':<30} {r2_physics:>8.4f} {mae_physics:>12.2f}")

# ── STEP 6: Save charts ───────────────────────────────────
print("\n[6] Saving charts...")

# Chart 1: Model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_xgb, alpha=0.3, s=10, color="#4C72B0", label="XGBoost")
axes[0].scatter(y_test, y_pred_physics, alpha=0.3, s=10, color="#55A868", label="+ Physics")
min_v = float(y_test.min())
max_v = float(y_test.max())
axes[0].plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Perfect")
axes[0].set_xlabel("Actual Bulk Modulus (GPa)")
axes[0].set_ylabel("Predicted Bulk Modulus (GPa)")
axes[0].set_title("XGBoost vs Physics-Corrected Predictions")
axes[0].legend()

axes[1].hist(df_check["nu_violation"], bins=50, color="#C44E52", edgecolor="white")
axes[1].axvline(0.05, color="black", linestyle="--", linewidth=2,
                label="Violation threshold (0.05)")
axes[1].set_title("Physics Violation: Derived vs Reported Poisson Ratio")
axes[1].set_xlabel("Absolute Difference")
axes[1].set_ylabel("Number of Materials")
axes[1].legend()

plt.tight_layout()
plt.savefig("notebooks/figures/physics_model_comparison.png", dpi=130)
plt.close()
print("    Saved: physics_model_comparison.png")

# Chart 2: Young's modulus distribution (derived from physics)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df_check["youngs_modulus_GPa"], bins=50, color="#8172B2", edgecolor="white")
ax.set_title("Young's Modulus Distribution (derived from E = 9KG/(3K+G))")
ax.set_xlabel("Young's Modulus (GPa)")
ax.set_ylabel("Number of Materials")
plt.tight_layout()
plt.savefig("notebooks/figures/youngs_modulus_distribution.png", dpi=130)
plt.close()
print("    Saved: youngs_modulus_distribution.png")

print("\n" + "=" * 60)
print("  Physics-informed model complete!")
print(f"  Best R²: {max(r2_xgb, r2_physics):.4f}")
print("=" * 60)