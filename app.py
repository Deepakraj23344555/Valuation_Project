import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="GT Valuation Analytics", page_icon="📊", layout="wide")

# Custom CSS for a professional "FinTech" look
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #f8f9fa; }
    
    /* Headings */
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #4B0082; }
    
    /* Button Styling */
    .stButton>button { 
        background-color: #4B0082; 
        color: white; 
        border-radius: 5px; 
        border: none;
    }
    .stButton>button:hover {
        background-color: #38006b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---

def generate_template():
    """Generates a dummy Excel file for the user."""
    data = {
        'Year': [2025, 2026, 2027, 2028, 2029],
        'Revenue': [1000, 1100, 1210, 1331, 1464],
        'EBITDA_Margin': [0.20, 0.22, 0.24, 0.25, 0.25],
        'D_and_A': [50, 55, 60, 65, 70],
        'CapEx': [60, 65, 70, 75, 80],
        'Change_in_NWC': [20, 22, 24, 26, 28]
    }
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Model_Input')
    return output.getvalue()

def calculate_wacc(rf, beta, erp, cost_debt, tax_rate, equity_weight, debt_weight):
    """Calculates WACC using CAPM."""
    cost_equity = rf + beta * (erp - rf) # CAPM Formula
    after_tax_cost_debt = cost_debt * (1 - tax_rate)
    wacc = (cost_equity * equity_weight) + (after_tax_cost_debt * debt_weight)
    return wacc, cost_equity

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Valuation Toolkit")
    st.markdown("---")
    nav = st.radio("Navigate", ["📂 Data Upload", "🧮 WACC Builder", "📊 DCF Analysis", "🎲 Risk Simulation", "📘 Methodology"])
    st.markdown("---")
    st.caption("v1.2 | Grant Thornton Live Project")

# --- 4. MAIN APP LOGIC ---

# --------------------------
# TAB 1: DATA UPLOAD
# --------------------------
if nav == "📂 Data Upload":
    st.title("📂 Data Management")
    st.markdown("Upload historical financials and projections to initialize the model.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Step 1: Get Template")
        st.write("Use this standardized format for your client data.")
        template = generate_template()
        st.download_button("📥 Download Excel Template", data=template, file_name="Client_Data_Template.xlsx")
    
    with col2:
        st.subheader("Step 2: Upload Data")
        uploaded_file = st.file_uploader("Upload Populated Excel", type=['xlsx'])
        
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.session_state['data'] = df # Save to session state
            st.success("✅ Data Loaded Successfully!")
            st.dataframe(df.head(), use_container_width=True)

# --------------------------
# TAB 2: WACC BUILDER
# --------------------------
elif nav == "🧮 WACC Builder":
    st.title("🧮 Weighted Average Cost of Capital (WACC)")
    st.markdown("Build your discount rate using the **Capital Asset Pricing Model (CAPM)**.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Equity Assumptions")
        rf = st.number_input("Risk-Free Rate (%)", 3.0, 10.0, 4.25) / 100
        beta = st.number_input("Beta (Risk)", 0.5, 3.0, 1.15)
        erp = st.number_input("Market Return (%)", 5.0, 15.0, 10.0) / 100
        
    with col2:
        st.subheader("Debt Assumptions")
        cost_debt = st.number_input("Pre-Tax Cost of Debt (%)", 2.0, 15.0, 6.5) / 100
        tax_rate = st.number_input("Corporate Tax Rate (%)", 15.0, 40.0, 25.0) / 100
        
    with col3:
        st.subheader("Capital Structure")
        equity_percent = st.slider("Equity %", 0, 100, 75) / 100
        debt_percent = 1 - equity_percent
        st.write(f"**Debt %:** {debt_percent:.0%}")
        
    # Calculate
    calc_wacc, cost_equity = calculate_wacc(rf, beta, erp, cost_debt, tax_rate, equity_percent, debt_percent)
    
    # Save WACC to session
    st.session_state['wacc'] = calc_wacc
    st.session_state['tax_rate'] = tax_rate
    
    st.divider()
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Calculated Cost of Equity", f"{cost_equity:.2%}")
    metric2.metric("After-Tax Cost of Debt", f"{cost_debt * (1-tax_rate):.2%}")
    metric3.metric("👉 Final WACC", f"{calc_wacc:.2%}", delta_color="normal")

# --------------------------
# TAB 3: DCF ANALYSIS
# --------------------------
elif nav == "📊 DCF Analysis":
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data in the 'Data Upload' tab first.")
    else:
        st.title("📊 Discounted Cash Flow (DCF) Valuation")
        
        # Inputs
        df = st.session_state['data'].copy()
        wacc = st.session_state.get('wacc', 0.10) # Default 10% if not calculated
        tax_rate = st.session_state.get('tax_rate', 0.25)
        
        st.sidebar.markdown("### DCF Controls")
        tgr = st.sidebar.slider("Terminal Growth Rate (%)", 1.0, 5.0, 2.5) / 100
        
        # --- LOGIC ---
        df['EBITDA'] = df['Revenue'] * df['EBITDA_Margin']
        df['EBIT'] = df['EBITDA'] - df['D_and_A']
        df['NOPAT'] = df['EBIT'] * (1 - tax_rate)
        df['UFCF'] = df['NOPAT'] + df['D_and_A'] - df['CapEx'] - df['Change_in_NWC']
        
        # Discounting
        df['Period'] = range(1, len(df) + 1)
        df['Discount_Factor'] = 1 / ((1 + wacc) ** df['Period'])
        df['PV_UFCF'] = df['UFCF'] * df['Discount_Factor']
        
        # Terminal Value
        last_ufcf = df['UFCF'].iloc[-1]
        tv = (last_ufcf * (1 + tgr)) / (wacc - tgr)
        pv_tv = tv * df['Discount_Factor'].iloc[-1]
        
        enterprise_value = df['PV_UFCF'].sum() + pv_tv
        
        # SAVE TO STATE (For Risk Tab)
        st.session_state['enterprise_value'] = enterprise_value
        
        # --- OUTPUT ---
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Implied Enterprise Value", f"${enterprise_value:,.0f}")
        kpi2.metric("PV of Terminal Value", f"${pv_tv:,.0f}", f"{(pv_tv/enterprise_value)*100:.1f}% of Total")
        kpi3.metric("WACC Used", f"{wacc:.2%}")
        
        # Visuals
        tab1, tab2 = st.tabs(["Waterfall Valuation", "Sensitivity Matrix (Heatmap)"])
        
        with tab1:
            fig = go.Figure(go.Waterfall(
                orientation = "v",
                measure = ["relative", "relative", "total"],
                x = ["PV of Explicit Cash Flows", "PV of Terminal Value", "Enterprise Value"],
                y = [df['PV_UFCF'].sum(), pv_tv, enterprise_value],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
                textposition = "outside",
                text = [f"${df['PV_UFCF'].sum():,.0f}", f"${pv_tv:,.0f}", f"${enterprise_value:,.0f}"]
            ))
            fig.update_layout(title="Enterprise Value Composition")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Sensitivity Analysis: EV based on WACC vs. Growth Rate")
            
            wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
            tgr_range = [tgr - 0.01, tgr - 0.005, tgr, tgr + 0.005, tgr + 0.01]
            
            sensitivity_data = []
            for w in wacc_range:
                row = []
                for g in tgr_range:
                    term_val = (last_ufcf * (1 + g)) / (w - g)
                    pv_term = term_val * (1 / ((1 + w) ** len(df)))
                    pv_explicit = (df['UFCF'] * (1 / ((1 + w) ** df['Period']))).sum()
                    row.append(pv_term + pv_explicit)
                sensitivity_data.append(row)
                
            fig_heat = px.imshow(sensitivity_data,
                                labels=dict(x="Terminal Growth Rate", y="WACC", color="Enterprise Value"),
                                x=[f"{x:.1%}" for x in tgr_range],
                                y=[f"{y:.1%}" for y in wacc_range],
                                text_auto='.2s', aspect="auto", color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_heat, use_container_width=True)

# --------------------------
# TAB 4: RISK SIMULATION (Updated)
# --------------------------
elif nav == "🎲 Risk Simulation":
    if 'enterprise_value' not in st.session_state:
        st.warning("⚠️ Please run the 'DCF Analysis' tab first to generate a base valuation.")
    else:
        st.title("🎲 Monte Carlo Simulation")
        st.markdown("**Objective:** Stress-test the valuation using stochastic modeling.")
        
        col1, col2 = st.columns(2)
        with col1:
            # Dropdown for iterations
            iterations = st.selectbox("Number of Iterations (N)", [1000, 5000, 10000, 50000], index=1)
            st.caption("Higher N = Higher Statistical Convergence")
        
        with col2:
            volatility = st.slider("Revenue Volatility (σ)", 5, 40, 15) / 100
            st.caption("Standard Deviation of asset value")

        st.divider()

        if st.button("▶️ Run Monte Carlo Simulation"):
            with st.spinner(f"Running {iterations:,} stochastic scenarios..."):
                
                # Logic: Geometric Brownian Motion Proxy
                base_ev = st.session_state['enterprise_value']
                mu = 0 
                sigma = volatility
                shocks = np.random.normal(mu, sigma, iterations)
                simulated_values = base_ev * (1 + shocks)
                
                # Metrics
                mean_val = np.mean(simulated_values)
                var_95 = np.percentile(simulated_values, 5) # 5th Percentile
                upside_95 = np.percentile(simulated_values, 95)
                
                # Scorecards
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Expected Value", f"${mean_val:,.0f}")
                m2.metric("Value at Risk (5%)", f"${var_95:,.0f}", delta_color="inverse")
                m3.metric("Upside Potential (95%)", f"${upside_95:,.0f}")
                
                # Histogram
                fig_hist = px.histogram(
                    simulated_values, 
                    nbins=75, 
                    title=f"Distribution of {iterations:,} Valuation Outcomes",
                    color_discrete_sequence=['#4B0082']
                )
                fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="black", annotation_text="Mean")
                fig_hist.add_vline(x=var_95, line_dash="dot", line_color="red", annotation_text="Risk Floor (5%)")
                st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------
# TAB 5: DOCUMENTATION
# --------------------------
elif nav == "📘 Methodology":
    st.title("📘 Valuation Methodology")
    st.markdown("""
    ### 1. Discounted Cash Flow (DCF)
    We utilize a 5-year explicit period forecast followed by a terminal value calculation using the Gordon Growth Method.
    
    $$
    Enterprise Value = \\sum_{t=1}^{5} \\frac{UFCF_t}{(1+WACC)^t} + \\frac{Terminal Value}{(1+WACC)^5}
    $$
    
    ### 2. WACC (Capital Asset Pricing Model)
    The discount rate is derived using the standard CAPM formula:
    
    $$
    K_e = R_f + \\beta (R_m - R_f)
    $$
    
    *Where $R_f$ is the Risk-Free Rate, $\\beta$ is the levered beta, and $(R_m - R_f)$ is the Market Risk Premium.*
    
    ### 3. Monte Carlo Simulation
    To account for forecasting uncertainty, we apply stochastic shocks to the revenue baseline following a normal distribution:
    
    $$
    EV_{sim} = EV_{base} \\times (1 + \\mathcal{N}(0, \\sigma))
    $$
    
    *Where $\\sigma$ represents the implied volatility of the sector.*
    """)
