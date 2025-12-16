import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="GT Valuation Analytics", page_icon="💼", layout="wide")

# --- EXECUTIVE THEME CSS (Fixed Visibility) ---
st.markdown("""
    <style>
    /* 1. Main Background - Professional Platinum Gray */
    .stApp { 
        background-color: #F0F2F6; 
    }
    
    /* 2. Sidebar Background - Dark Navy */
    section[data-testid="stSidebar"] {
        background-color: #111827; /* Very Dark Navy */
    }
    
    /* 3. Text Visibility Fix - Global */
    h1, h2, h3, h4, h5, h6 {
        color: #1F2937 !important; /* Dark Charcoal for Headings */
        font-family: 'Helvetica Neue', sans-serif;
    }
    p, div, label, span {
        color: #374151; /* Standard Gray for body text */
    }
    
    /* 4. Sidebar Text Color - Override to White */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p {
        color: #F9FAFB !important; /* White/Light Gray */
    }
    
    /* 5. Metric Cards - High Contrast */
    div[data-testid="stMetricValue"] { 
        font-size: 28px; 
        color: #4B0082; /* GT Purple */
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #4B5563; /* Dark Gray */
    }
    
    /* 6. Professional Buttons */
    .stButton>button {
        background-color: #4B0082; /* GT Purple */
        color: white !important;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
        width: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { 
        background-color: #38006b; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 7. Containers/Tabs (White Cards on Gray Background) */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #FFFFFF; 
        border-radius: 4px 4px 0 0; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
        padding: 10px 20px;
        color: #374151; /* Text inside tabs */
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
    cost_equity = rf + beta * (erp - rf) 
    after_tax_cost_debt = cost_debt * (1 - tax_rate)
    wacc = (cost_equity * equity_weight) + (after_tax_cost_debt * debt_weight)
    return wacc, cost_equity

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("💼 Valuation Engine") 
    st.caption("Grant Thornton | Live Project v1.3")
    st.markdown("---")
    
    # Navigation Radio Button
    nav = st.radio("Navigation Module", 
        ["🗂️ Data Manager", 
         "⚖️ Cost of Capital (WACC)", 
         "💎 Intrinsic Valuation", 
         "⚡ Stress Testing (VaR)", 
         "📝 Assumptions & Logic"])
    
    st.markdown("---")
    st.info("💡 **Analyst Note:** Ensure all inputs are annual figures in USD Millions.")

# --- 4. MAIN APP LOGIC ---

# --------------------------
# TAB 1: DATA MANAGER
# --------------------------
if nav == "🗂️ Data Manager":
    st.title("🗂️ Data Management")
    st.markdown("Initialize the model by importing client financial data.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Template Access")
        st.write("Standardized template for financial ingestion.")
        template = generate_template()
        st.download_button("💾 Download Standard Template", data=template, file_name="GT_Client_Template.xlsx")
    
    with col2:
        st.subheader("2. Data Ingestion")
        uploaded_file = st.file_uploader("Import Excel File", type=['xlsx'])
        
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.session_state['data'] = df 
            st.success("✅ Data Successfully Ingested")
            with st.expander("👁️ View Raw Data"):
                st.dataframe(df, use_container_width=True)

# --------------------------
# TAB 2: WACC BUILDER
# --------------------------
elif nav == "⚖️ Cost of Capital (WACC)":
    st.title("⚖️ Cost of Capital Builder")
    st.markdown("Configure the discount rate using the **Capital Asset Pricing Model (CAPM)**.")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Equity Parameters") 
        rf = st.number_input("Risk-Free Rate (Rf) %", 3.0, 10.0, 4.25) / 100
        beta = st.number_input("Levered Beta (β)", 0.5, 3.0, 1.15)
        erp = st.number_input("Market Risk Premium (Rm-Rf) %", 5.0, 15.0, 10.0) / 100
        
    with col2:
        st.markdown("#### Debt Parameters")
        cost_debt = st.number_input("Pre-Tax Cost of Debt (Kd) %", 2.0, 15.0, 6.5) / 100
        tax_rate = st.number_input("Marginal Tax Rate (t) %", 15.0, 40.0, 25.0) / 100
        
    with col3:
        st.markdown("#### Capital Structure")
        equity_percent = st.slider("Equity %", 0, 100, 75) / 100
        debt_percent = 1 - equity_percent
        st.write(f"**Debt %:** {debt_percent:.0%}")
        
    # Calculate Logic
    calc_wacc, cost_equity = calculate_wacc(rf, beta, erp, cost_debt, tax_rate, equity_percent, debt_percent)
    
    # Save WACC to session
    st.session_state['wacc'] = calc_wacc
    st.session_state['tax_rate'] = tax_rate
    
    st.markdown("---")
    st.subheader("Results")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Cost of Equity (Ke)", f"{cost_equity:.2%}")
    metric2.metric("Post-Tax Cost of Debt (Kd)", f"{cost_debt * (1-tax_rate):.2%}")
    metric3.metric("👉 Final WACC", f"{calc_wacc:.2%}", delta_color="normal")

# --------------------------
# TAB 3: DCF ANALYSIS
# --------------------------
elif nav == "💎 Intrinsic Valuation":
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data in the 'Data Manager' module first.")
    else:
        st.title("💎 Discounted Cash Flow (DCF)")
        
        # Inputs
        df = st.session_state['data'].copy()
        wacc = st.session_state.get('wacc', 0.10) # Default 10% if not calculated
        tax_rate = st.session_state.get('tax_rate', 0.25)
        
        st.sidebar.markdown("### 💎 Valuation Controls")
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
        kpi3.metric("Discount Rate (WACC)", f"{wacc:.2%}")
        
        st.divider()
        
        # Visuals
        tab1, tab2 = st.tabs(["Waterfall Breakdown", "Sensitivity Heatmap"])
        
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
            fig.update_layout(title="Enterprise Value Composition", template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
            fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat, use_container_width=True)

# --------------------------
# TAB 4: RISK SIMULATION (Updated)
# --------------------------
elif nav == "⚡ Stress Testing (VaR)":
    if 'enterprise_value' not in st.session_state:
        st.warning("⚠️ Please run the 'Intrinsic Valuation' module first.")
    else:
        st.title("⚡ Value at Risk (VaR) Analysis")
        st.markdown("**Objective:** Stress-test the valuation using stochastic modeling.")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            iterations = st.selectbox("Simulation Iterations (N)", [1000, 5000, 10000, 50000], index=1)
            st.caption("Statistical Significance increases with N.")
        
        with col2:
            volatility = st.slider("Implied Volatility (σ)", 5, 40, 15) / 100
            st.caption("Standard Deviation based on historical peer volatility.")

        if st.button("▶️ Execute Monte Carlo Simulation"):
            with st.spinner(f"Computing {iterations:,} scenarios..."):
                
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
                m2.metric("Risk Floor (95% Conf.)", f"${var_95:,.0f}", delta_color="inverse")
                m3.metric("Upside Ceiling (95% Conf.)", f"${upside_95:,.0f}")
                
                # Histogram
                fig_hist = px.histogram(
                    simulated_values, 
                    nbins=75, 
                    title=f"Probability Distribution of Valuation Outcomes (N={iterations:,})",
                    color_discrete_sequence=['#4B0082']
                )
                fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="black", annotation_text="Mean")
                fig_hist.add_vline(x=var_95, line_dash="dot", line_color="red", annotation_text="VaR (5%)")
                
                fig_hist.update_layout(template="plotly_white", xaxis_title="Enterprise Value ($)", yaxis_title="Frequency", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------
# TAB 5: DOCUMENTATION
# --------------------------
elif nav == "📝 Assumptions & Logic":
    st.title("📝 Methodology Note")
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
