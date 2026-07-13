# explore.py
# This script loads your 6 datasets and prints basic info about each one.
# Run it with: python explore.py

import pandas as pd

print("Loading your datasets...")
print("=" * 50)

# Dataset 1: Material Properties
print("\n[1] MATERIAL PROPERTIES (DS1)")
df1 = pd.read_csv("data/raw/DS1_material_properties_5500.csv")
print(f"  Rows: {len(df1)}")
print(f"  Columns: {df1.shape[1]}")
print(f"  Crystal systems: {df1['crystal_system'].unique().tolist()}")

# Dataset 2: Commodity Prices
print("\n[2] COMMODITY PRICES (DS2)")
df2 = pd.read_csv("data/raw/DS2_commodity_prices_10yr.csv")
print(f"  Rows: {len(df2)}")
print(f"  Commodities: {df2['commodity'].unique().tolist()}")
print(f"  Date range: {df2['date'].min()} to {df2['date'].max()}")

# Dataset 3: Infrastructure Bridges
print("\n[3] INFRASTRUCTURE BRIDGES (DS3)")
df3 = pd.read_csv("data/raw/DS3_infrastructure_bridges_5000.csv")
print(f"  Rows: {len(df3)}")
print(f"  Bridge materials: {df3['material'].unique().tolist()}")
print(f"  Deficient bridges: {df3['structurally_deficient'].sum()}")

# Dataset 4: Cross-Domain Features
print("\n[4] CROSS-DOMAIN FEATURES (DS4)")
df4 = pd.read_csv("data/raw/DS4_crossdomain_features_daily.csv")
print(f"  Rows: {len(df4)}")
print(f"  Columns: {list(df4.columns)}")

# Dataset 5: Element Prices
print("\n[5] ELEMENT PRICES (DS5)")
df5 = pd.read_csv("data/raw/DS5_element_prices_monthly.csv")
print(f"  Rows: {len(df5)}")
print(f"  Elements tracked: {df5['element'].nunique()}")
top3 = df5.groupby("element")["price_usd_per_kg"].mean().sort_values(ascending=False).head(3)
print(f"  Top 3 most expensive elements:")
for elem, price in top3.items():
    print(f"    {elem}: ${price:,.0f} per kg")

# Dataset 6: Historical Failures
print("\n[6] HISTORICAL FAILURES (DS6)")
df6 = pd.read_csv("data/raw/DS6_historical_failures_2000.csv")
print(f"  Rows: {len(df6)}")
print(f"  Failure modes: {df6['failure_mode'].unique().tolist()}")
print(f"  Predicted ahead of time: {df6['was_predicted'].mean():.1%}")

print("\n" + "=" * 50)
print("All 6 datasets loaded successfully!")
print("You are ready to start building MatRisk AI!")