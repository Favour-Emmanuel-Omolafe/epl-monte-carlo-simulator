import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

st.set_page_config(page_title="EPL 15-Year Monte Carlo Engine", page_icon="⚽", layout="wide")

st.title("⚽ Premier League: 15-Year Monte Carlo Simulation Engine")
st.markdown("*Simulating Premier League dynasties (2026–2041) using Poisson goal models, squad depth, and financial shocks.*")

st.sidebar.header("🛠️ Simulation Controls")
num_simulations = st.sidebar.slider("Number of 15-Year Simulations", min_value=50, max_value=300, value=100, step=50)

selected_boost_club = st.sidebar.selectbox("Select a Club to Boost:", ["None", "Chelsea", "Arsenal", "Liverpool", "Man City", "Newcastle", "Aston Villa", "Brentford"])
budget_boost = st.sidebar.slider(f"Transfer Power Boost for {selected_boost_club} (%)", min_value=-30, max_value=50, value=0, step=5)
enable_crises = st.sidebar.checkbox("Enable PSR/SCR Point Deductions", value=True)

# Teams & Ratings
teams = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich', 
    'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle', 
    'Nott\'m Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
]

base_ratings = pd.DataFrame({
    'home_attack': [1.35, 1.15, 0.95, 1.05, 1.10, 1.25, 0.95, 0.85, 0.95, 0.80, 0.85, 1.30, 1.45, 1.10, 1.15, 0.90, 0.75, 1.15, 0.95, 0.85],
    'home_defense': [0.70, 0.95, 1.10, 1.05, 1.00, 0.85, 1.05, 1.15, 1.05, 1.25, 1.20, 0.75, 0.65, 1.00, 0.95, 1.15, 1.30, 1.00, 1.10, 1.15],
    'away_attack': [1.25, 1.05, 0.85, 0.95, 1.00, 1.15, 0.85, 0.75, 0.85, 0.70, 0.75, 1.20, 1.35, 1.00, 1.05, 0.80, 0.65, 1.05, 0.85, 0.75],
    'away_defense': [0.75, 1.00, 1.15, 1.10, 1.05, 0.90, 1.10, 1.20, 1.10, 1.30, 1.25, 0.80, 0.70, 1.05, 1.00, 1.20, 1.35, 1.05, 1.15, 1.20]
}, index=teams)

if selected_boost_club != "None" and budget_boost != 0:
    mult = 1.0 + (budget_boost / 100.0)
    base_ratings.loc[selected_boost_club, 'home_attack'] *= mult
    base_ratings.loc[selected_boost_club, 'away_attack'] *= mult

def sim_match(h, a, r):
    lh = max(0.1, r.loc[h, 'home_attack'] * r.loc[a, 'away_defense'] * 1.55)
    la = max(0.1, r.loc[a, 'away_attack'] * r.loc[h, 'home_defense'] * 1.25)
    return np.random.poisson(lh), np.random.poisson(la)

def sim_season(r, deds):
    t_list = list(r.index)
    tbl = {t: {'points': -deds.get(t, 0), 'gd': 0} for t in t_list}
    for h in t_list:
        for a in t_list:
            if h == a: continue
            hg, ag = sim_match(h, a, r)
            tbl[h]['gd'] += (hg - ag)
            tbl[a]['gd'] += (ag - hg)
            if hg > ag: tbl[h]['points'] += 3
            elif hg < ag: tbl[a]['points'] += 3
            else:
                tbl[h]['points'] += 1
                tbl[a]['points'] += 1
    df = pd.DataFrame.from_dict(tbl, orient='index').sort_values(by=['points', 'gd'], ascending=False)
    df['pos'] = range(1, len(t_list) + 1)
    return df

def sim_15_yrs(r, cr_on):
    cur_r = r.copy()
    champs = []
    top4 = []
    for _ in range(15):
        deds = {}
        if cr_on and np.random.rand() < 0.15:
            target = np.random.choice(cur_r.index)
            deds[target] = 6
            cur_r.loc[target, 'home_attack'] *= 0.92
        tbl = sim_season(cur_r, deds)
        champs.append(tbl.index[0])
        top4.extend(list(tbl.index[:4]))
        for t in cur_r.index:
            if tbl.loc[t, 'pos'] <= 4:
                cur_r.loc[t, 'home_attack'] *= 1.025
            else:
                cur_r.loc[t, 'home_attack'] *= np.random.uniform(0.98, 1.015)
    return champs, top4

if st.button("🚀 Run 15-Year Simulation"):
    with st.spinner(f"Simulating {num_simulations} universes..."):
        champ_counts = defaultdict(int)
        top4_counts = defaultdict(int)
        for _ in range(num_simulations):
            c_list, t_list = sim_15_yrs(base_ratings, enable_crises)
            for c in c_list: champ_counts[c] += 1
            for t in t_list: top4_counts[t] += 1

        tot = num_simulations * 15
        out = [{'Club': t, 'Titles Won': champ_counts[t], 'Title Share (%)': round((champ_counts[t]/tot)*100, 1), 'Top 4 (%)': round((top4_counts[t]/tot)*100, 1)} for t in base_ratings.index]
        res_df = pd.DataFrame(out).sort_values(by='Titles Won', ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("📊 Projected Title Share")
        top_c = res_df[res_df['Titles Won'] > 0].sort_values(by='Titles Won', ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(top_c['Club'], top_c['Title Share (%)'], color='#034694', edgecolor='black')
        for b in bars:
            w = b.get_width()
            ax.text(w + 0.3, b.get_y() + b.get_height()/2, f'{w:.1f}%', va='center', fontsize=9, fontweight='bold')
        ax.set_xlim(0, max(top_c['Title Share (%)']) * 1.25)
        ax.set_xlabel("Title Share (%)")
        st.pyplot(fig)
    with c2:
        st.subheader("📋 Leaderboard")
        st.dataframe(res_df.head(10), use_container_width=True)
else:
    st.info("Click 'Run 15-Year Simulation' to compute.")
