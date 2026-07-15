# write_app.py - writes the complete fixed dashboard
code = open("src/dashboard/app.py", "w", encoding="utf-8")
code.write("""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MatRisk AI", page_icon="🔬", layout="wide")

@st.cache_data
def load_all_data():
    df1 = pd.read_csv("data/raw/DS1_material_properties_5500.csv")
    df2 = pd.read_csv("data/raw/DS2_commodity_prices_10yr.csv", parse_dates=["date"])
    df3 = pd.read_csv("data/raw/DS3_infrastructure_bridges_5000.csv")
    df4 = pd.read_csv("data/raw/DS4_crossdomain_features_daily.csv", parse_dates=["date"])
    df6 = pd.read_csv("data/raw/DS6_historical_failures_2000.csv")
    return df1, df2, df3, df4, df6

@st.cache_resource
def train_model(df1):
    le_crystal  = LabelEncoder()
    le_category = LabelEncoder()
    df = df1.copy()
    df["crystal_system_code"] = le_crystal.fit_transform(df["crystal_system"])
    df["category_code"]       = le_category.fit_transform(df["category"])
    feature_cols = [
        "n_elements","crystal_system_code","category_code",
        "formation_energy_per_atom_eV","energy_above_hull_eV",
        "band_gap_eV","is_metal","shear_modulus_GPa",
        "poisson_ratio","density_g_cm3","nsites","melting_point_K",
    ]
    df_clean = df[feature_cols + ["bulk_modulus_GPa"]].dropna()
    mdl = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    mdl.fit(df_clean[feature_cols], df_clean["bulk_modulus_GPa"])
    return mdl, le_crystal, le_category, feature_cols

with st.spinner("Loading MatRisk AI..."):
    df1, df2, df3, df4, df6 = load_all_data()
    model, le_crystal, le_category, feature_cols = train_model(df1)

st.sidebar.title("🔬 MatRisk AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to", [
    "🏠 Home",
    "🔬 Material Predictor",
    "📈 Commodity Signals",
    "🌉 Bridge Risk",
    "🌱 ESG Scoring",
    "🎮 MatRisk Lab"
])

if page == "🏠 Home":
    st.title("🔬 MatRisk AI Dashboard")
    st.markdown("Connecting Material Science with Financial Risk Intelligence")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔬 Materials",    f"{len(df1):,}")
    c2.metric("📈 Price Records",f"{len(df2):,}")
    c3.metric("🌉 Bridges",      f"{len(df3):,}")
    c4.metric("💥 Failures",     f"{len(df6):,}")
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Crystal System Distribution")
        counts = df1["crystal_system"].value_counts().reset_index()
        counts.columns = ["Crystal System","Count"]
        fig = px.bar(counts, x="Crystal System", y="Count",
                     color="Count", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader("Commodity Price History")
        sel = st.selectbox("Pick commodity", df2["commodity"].unique())
        df_c = df2[df2["commodity"]==sel].sort_values("date")
        fig2 = px.line(df_c, x="date", y="close")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.subheader("Model Performance Summary")
    perf = pd.DataFrame({
        "Model":  ["Material Predictor","Physics Model","Commodity Predictor",
                   "Bridge Survival","Insurance Model","ESG Scoring"],
        "Result": ["R2=0.9858","R2=0.9853","57.2% accuracy",
                   "Cox PH fitted","99th VaR=108M","Steel top 68.2"],
        "Status": ["Done","Done","Done","Done","Done","Done"]
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

elif page == "🔬 Material Predictor":
    st.title("🔬 Material Property Predictor")
    st.info("Adjust the sliders and click Predict!")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        crystal_system   = st.selectbox("Crystal System", df1["crystal_system"].unique())
        category         = st.selectbox("Category", df1["category"].unique())
        n_elements       = st.slider("Number of Elements", 1, 8, 2)
        is_metal         = st.radio("Is Metal?", [1,0], format_func=lambda x: "Yes" if x else "No")
        formation_energy = st.slider("Formation Energy (eV/atom)", -8.0, 2.0, -1.0, 0.1)
        energy_hull      = st.slider("Energy Above Hull (eV)", 0.0, 1.0, 0.0, 0.01)
    with col2:
        band_gap      = st.slider("Band Gap (eV)", 0.0, 12.0, 1.0, 0.1)
        shear_modulus = st.slider("Shear Modulus (GPa)", 0.0, 400.0, 50.0, 1.0)
        poisson_ratio = st.slider("Poisson Ratio", -0.5, 0.5, 0.25, 0.01)
        density       = st.slider("Density (g/cm3)", 0.5, 25.0, 5.0, 0.1)
        nsites        = st.slider("Number of Sites", 1, 50, 4)
        melting_point = st.slider("Melting Point (K)", 200.0, 4000.0, 1500.0, 10.0)
    st.markdown("---")
    if st.button("Predict Bulk Modulus", type="primary"):
        try:
            cc = le_crystal.transform([crystal_system])[0]
            kc = le_category.transform([category])[0]
        except Exception:
            cc, kc = 0, 0
        inp = pd.DataFrame([[
            n_elements, cc, kc, formation_energy, energy_hull,
            band_gap, is_metal, shear_modulus, poisson_ratio,
            density, nsites, melting_point
        ]], columns=feature_cols)
        pred = model.predict(inp)[0]
        if abs(1 - 2*poisson_ratio) > 1e-6:
            K_phys   = 2*shear_modulus*(1+poisson_ratio)/(3*(1-2*poisson_ratio))
            phys_err = abs(pred - K_phys)
        else:
            K_phys, phys_err = None, 0
        r1,r2,r3 = st.columns(3)
        r1.metric("Predicted", str(round(pred,1)) + " GPa")
        if K_phys:
            r2.metric("Physics Expected", str(round(K_phys,1)) + " GPa")
        r3.metric("Physics Check", "OK" if phys_err < 20 else "Review")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pred,
            title={"text":"Bulk Modulus (GPa)"},
            gauge={"axis":{"range":[0,500]},
                   "bar":{"color":"#1f77b4"},
                   "steps":[{"range":[0,100],"color":"#d4edda"},
                             {"range":[100,250],"color":"#fff3cd"},
                             {"range":[250,500],"color":"#f8d7da"}]}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        label = ("Very Stiff" if pred>300 else "Stiff" if pred>150
                 else "Moderate" if pred>70 else "Soft")
        st.success(str(round(pred,1)) + " GPa — " + label)

elif page == "📈 Commodity Signals":
    st.title("📈 Commodity Price Signals")
    st.markdown("---")
    df_merged = pd.merge(df2, df4, on=["date","commodity"], how="inner")
    commodity = st.selectbox("Select Commodity", df2["commodity"].unique())
    df_c = df_merged[df_merged["commodity"]==commodity].sort_values("date")
    col_chart, col_sig = st.columns([3,1])
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_c["date"], y=df_c["close"],
            name="Price", line=dict(color="#1f77b4",width=1.5)))
        fig.add_trace(go.Scatter(x=df_c["date"], y=df_c["sma_21"],
            name="21d MA", line=dict(color="#ff7f0e",dash="dash")))
        fig.add_trace(go.Scatter(x=df_c["date"], y=df_c["bollinger_upper"],
            name="BB Upper", line=dict(color="#2ca02c",dash="dot")))
        fig.add_trace(go.Scatter(x=df_c["date"], y=df_c["bollinger_lower"],
            name="BB Lower", line=dict(color="#d62728",dash="dot")))
        fig.update_layout(title=commodity + " Price and Indicators",
                          height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with col_sig:
        st.subheader("Live Signals")
        latest   = df_c.iloc[-1]
        rsi_val  = float(latest["rsi_14"])
        bz_val   = float(latest["bollinger_z"])
        macd_val = float(latest["macd"])
        mqi_val  = float(latest["mqi"])
        sdp_val  = float(latest["supply_disruption_prob"])
        rsi_delta = "Overbought" if rsi_val>70 else "Oversold" if rsi_val<30 else "Neutral"
        st.metric("RSI (14)",    str(round(rsi_val,1)),  rsi_delta)
        st.metric("Bollinger Z", str(round(bz_val,2)),   "Above avg" if bz_val>0 else "Below avg")
        st.metric("MACD",        str(round(macd_val,2)), "Bullish" if macd_val>0 else "Bearish")
        st.metric("MQI",         str(round(mqi_val,3)))
        if sdp_val>0.3:
            st.error("High supply risk: " + str(round(sdp_val*100,1)) + "%")
        elif sdp_val>0.15:
            st.warning("Medium supply risk: " + str(round(sdp_val*100,1)) + "%")
        else:
            st.success("Low supply risk: " + str(round(sdp_val*100,1)) + "%")

elif page == "🌉 Bridge Risk":
    st.title("🌉 Bridge Infrastructure Risk")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Bridges", str(len(df3)))
    deficient = int(df3["structurally_deficient"].sum())
    c2.metric("Deficient", str(deficient))
    avg_age = round(float(df3["age_years"].mean()),1)
    c3.metric("Avg Age", str(avg_age) + " yrs")
    total_loans = round(float(df3["loan_outstanding_M"].sum()),0)
    c4.metric("Total Loans", "$" + str(total_loans) + "M")
    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        cond = df3["condition_rating"].value_counts().sort_index().reset_index()
        cond.columns = ["Rating","Count"]
        fig = px.bar(cond, x="Rating", y="Count",
                     color="Rating", color_continuous_scale="RdYlGn",
                     title="NBI Ratings (9=Excellent, 1=Critical)")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        df3_plot = df3.copy()
        df3_plot["Status"] = df3_plot["structurally_deficient"].map({0:"OK",1:"Deficient"})
        fig2 = px.scatter(df3_plot, x="age_years", y="corrosion_rate_mm_yr",
                          color="Status",
                          color_discrete_map={"OK":"#2ca02c","Deficient":"#d62728"},
                          title="Age vs Corrosion Rate")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.subheader("Individual Bridge Assessment")
    bridge_id = st.selectbox("Select Bridge", df3["bridge_id"].tolist())
    b = df3[df3["bridge_id"]==bridge_id].iloc[0]
    a, bb, c = st.columns(3)
    with a:
        st.metric("Material",  str(b["material"]))
        st.metric("Age",       str(b["age_years"]) + " yrs")
        st.metric("Condition", str(b["condition_rating"]) + "/9")
    with bb:
        life_pct = round(float(b["age_years"])/float(b["design_life_years"])*100,1)
        st.metric("Corrosion", str(round(float(b["corrosion_rate_mm_yr"]),4)) + " mm/yr")
        st.metric("Life Used", str(life_pct) + "%")
        st.metric("Loan",      "$" + str(round(float(b["loan_outstanding_M"]),1)) + "M")
    with c:
        lp = float(b["age_years"])/float(b["design_life_years"])
        risk = ((9-float(b["condition_rating"]))/9*40 +
                min(lp,1)*25 +
                min(float(b["corrosion_rate_mm_yr"])/0.15,1)*20 +
                (1-min(float(b["remaining_thickness_mm"])/float(b["original_thickness_mm"]),1))*15)
        rating = ("AAA" if risk<15 else "BBB" if risk<25
                  else "BB" if risk<40 else "B" if risk<55 else "CCC")
        st.metric("Risk Score",    str(round(risk,1)) + "/100")
        st.metric("Credit Rating", rating)
        st.metric("Risk Premium",  str(round(risk*5,0)) + " bps")

elif page == "🌱 ESG Scoring":
    st.title("🌱 ESG Commodity Scoring")
    st.markdown("---")
    latest_esg = df4.sort_values("date").groupby("commodity").last().reset_index()
    mx_c = float(latest_esg["carbon_intensity_virgin"].max())
    mx_g = float(latest_esg["green_premium_per_kg"].max())
    mx_d = float(latest_esg["supply_disruption_prob"].max())
    mx_h = float(latest_esg["herfindahl_index"].max())
    mx_s = float(latest_esg["substitution_elasticity"].max())
    latest_esg["E"] = ((1-latest_esg["carbon_intensity_virgin"]/mx_c)*60 +
                       (latest_esg["green_premium_per_kg"]/mx_g)*40).round(1)
    latest_esg["S"] = ((1-latest_esg["supply_disruption_prob"]/mx_d)*50 +
                       (1-latest_esg["herfindahl_index"]/mx_h)*50).round(1)
    latest_esg["G"] = (latest_esg["substitution_elasticity"]/mx_s*100).round(1)
    latest_esg["ESG"] = (latest_esg["E"]*0.4+latest_esg["S"]*0.35+latest_esg["G"]*0.25).round(1)
    esg_table = latest_esg[["commodity","E","S","G","ESG"]].sort_values("ESG",ascending=False)
    st.dataframe(esg_table, use_container_width=True, hide_index=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(esg_table.sort_values("ESG"), x="ESG", y="commodity",
                     orientation="h", color="ESG",
                     color_continuous_scale="RdYlGn", title="ESG Ranking")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(esg_table, x="commodity", y=["E","S","G"],
                      barmode="group", title="ESG Components",
                      color_discrete_map={"E":"#2ca02c","S":"#1f77b4","G":"#9467bd"})
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

elif page == "🎮 MatRisk Lab":
    st.title("🎮 MatRisk Lab Game")
    st.markdown("Manage your infrastructure portfolio!")
    st.markdown("---")
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {
            "cash":500,"bridges":3,"score":0,
            "turn":1,"events":[],"total_losses":0
        }
    p = st.session_state.portfolio
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Cash",    "$" + str(p["cash"]) + "M")
    c2.metric("Bridges", str(p["bridges"]))
    c3.metric("Score",   str(p["score"]))
    c4.metric("Turn",    str(p["turn"]))
    st.markdown("---")
    shocks = [
        {"name":"Corrosion Spike","desc":"Heavy rain accelerated corrosion.","loss":15,"type":"bad"},
        {"name":"Steel Price Drop","desc":"Repair costs reduced!","loss":-10,"type":"good"},
        {"name":"Flood Damage","desc":"Flash flood damaged foundations.","loss":25,"type":"bad"},
        {"name":"All Clear","desc":"All bridges passed inspection!","loss":0,"type":"neutral"},
        {"name":"New Alloy Discovery","desc":"Maintenance costs reduced.","loss":-20,"type":"good"},
        {"name":"Government Grant","desc":"Grant received!","loss":-30,"type":"good"},
        {"name":"Fatigue Failure","desc":"Bridge closed for repairs.","loss":40,"type":"bad"},
        {"name":"Supply Disruption","desc":"Material costs raised.","loss":12,"type":"bad"},
    ]
    col_game, col_log = st.columns([2,1])
    with col_game:
        if st.button("Draw Random Event", type="primary"):
            shock = shocks[np.random.randint(len(shocks))]
            p["cash"]         -= shock["loss"]
            p["total_losses"] += max(shock["loss"],0)
            p["score"]        += 10 if shock["type"]=="good" else -5 if shock["type"]=="bad" else 2
            p["turn"]         += 1
            p["events"].append("T" + str(p["turn"]-1) + ": " + shock["name"])
            st.session_state.portfolio = p
            if shock["type"] == "bad":
                st.error(shock["name"] + " — " + shock["desc"] + " Cost: $" + str(shock["loss"]) + "M")
            elif shock["type"] == "good":
                st.success(shock["name"] + " — " + shock["desc"] + " Gained: $" + str(abs(shock["loss"])) + "M")
            else:
                st.info(shock["name"] + " — " + shock["desc"])
        st.markdown("---")
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Repair Bridge (-$20M, +15pts)"):
                if p["cash"] >= 20:
                    p["cash"] -= 20
                    p["score"] += 15
                    st.success("Repaired! +15 pts")
                else:
                    st.error("Not enough cash!")
                st.session_state.portfolio = p
            if st.button("Build Bridge (-$80M, +50pts)"):
                if p["cash"] >= 80:
                    p["cash"] -= 80
                    p["bridges"] += 1
                    p["score"] += 50
                    st.success("Built! +50 pts")
                else:
                    st.error("Not enough cash!")
                st.session_state.portfolio = p
        with a2:
            if st.button("Run Inspection (-$5M, +8pts)"):
                if p["cash"] >= 5:
                    p["cash"] -= 5
                    p["score"] += 8
                    st.info("Done! +8 pts")
                else:
                    st.error("Not enough cash!")
                st.session_state.portfolio = p
            if st.button("Buy Insurance (-$10M, +5pts)"):
                if p["cash"] >= 10:
                    p["cash"] -= 10
                    p["score"] += 5
                    st.success("Insured!")
                else:
                    st.error("Not enough cash!")
                st.session_state.portfolio = p
        if st.button("Reset Game"):
            st.session_state.portfolio = {
                "cash":500,"bridges":3,"score":0,
                "turn":1,"events":[],"total_losses":0
            }
            st.rerun()
    with col_log:
        st.subheader("Event Log")
        for ev in reversed(p["events"][-10:]):
            st.caption(ev)
        st.markdown("---")
        health = int(min(max(p["cash"]/500*100,0),100))
        st.progress(health)
        st.caption("Cash health: " + str(health) + "%")
        if p["cash"] < 0:
            st.error("BANKRUPT!")
        elif p["cash"] < 50:
            st.error("Critical - very low cash!")
        elif p["cash"] < 150:
            st.warning("Low cash - be careful")
        else:
            st.success("Portfolio healthy")
""")
code.close()
print("App written successfully!")