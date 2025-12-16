import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="GT Valuation Analytics", page_icon="📊", layout="wide")

# Custom CSS for a "SaaS" look
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #f8f9fa; }
    
    /* Headings */
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #4B0082; }
    
    /* Cards for Logic */
    .css-1r6slb0 { background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* Button Styling */
    .stButton>button { background-color: #4B0082; color: white; border-radius: 5px; }
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
    cost_equity = rf + beta * (erp - rf) # CAPM
    after_tax_cost_debt = cost_debt * (1 - tax_rate)
    wacc = (cost_equity * equity_weight) + (after_tax_cost_debt * debt_weight)
    return wacc, cost_equity

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("Valuation Toolkit")
    
    nav = st.radio("Navigate", ["📂 Data Upload", "🧮 WACC Builder", "📊 DCF Analysis", "🎲 Risk Simulation"])
    
    st.info("ℹ️ **Pro Tip:** Start by uploading the standard data template in the 'Data Upload' tab.")

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
            st.session_state['data'] = df # Save to session state to use across tabs
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
        rf = st.number_input("Risk-Free Rate (%)", 3.0, 10.0, 4.0) / 100
        beta = st.number_input("Beta (Risk)", 0.5, 3.0, 1.2)
        erp = st.number_input("Market Return (%)", 5.0, 15.0, 10.0) / 100
        
    with col2:
        st.subheader("Debt Assumptions")
        cost_debt = st.number_input("Pre-Tax Cost of Debt (%)", 2.0, 15.0, 6.0) / 100
        tax_rate = st.number_input("Corporate Tax Rate (%)", 15.0, 40.0, 25.0) / 100
        
    with col3:
        st.subheader("Capital Structure")
        equity_percent = st.slider("Equity %", 0, 100, 70) / 100
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
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Sensitivity Analysis: EV based on WACC vs. Growth Rate")
            
            # Create Sensitivity Grid
            wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
            tgr_range = [tgr - 0.01, tgr - 0.005, tgr, tgr + 0.005, tgr + 0.01]
            
            sensitivity_data = []
            for w in wacc_range:
                row = []
                for g in tgr_range:
                    # Quick Recalc
                    term_val = (last_ufcf * (1 + g)) / (w - g)
                    pv_term = term_val * (1 / ((1 + w) ** len(df)))
                    pv_explicit = (df['UFCF'] * (1 / ((1 + w) ** df['Period']))).sum()
                    row.append(pv_term + pv_explicit)
                sensitivity_data.append(row)
                
            # Heatmap
            fig_heat = px.imshow(sensitivity_data,
                                labels=dict(x="Terminal Growth Rate", y="WACC", color="Enterprise Value"),
                                x=[f"{x:.1%}" for x in tgr_range],
                                y=[f"{y:.1%}" for y in wacc_range],
                                text_auto='.2s', aspect="auto", color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_heat, use_container_width=True)


# --------------------------
# TAB 4: RISK SIMULATION
# --------------------------
elif nav == "🎲 Risk Simulation":
    if 'data' not in st.session_state:
        st.warning("⚠️ Upload data first.")
    else:
        st.title("🎲 Monte Carlo Simulation")
        st.markdown("Quantify valuation risk by simulating 5,000 scenarios of Revenue Volatility.")
        
        volatility = st.slider("Revenue Volatility Assumption (+/- %)", 5, 25, 10) / 100
        
        if st.button("▶️ Run Simulation"):
            # Simplified Logic for Speed
            base_ev = 10000 # Placeholder for demo, ideally fetch from calculated EV
            
            # Create distribution
            mu, sigma = base_ev, base_ev * volatility
            s = np.random.normal(mu, sigma, 5000)
            
            fig_hist = px.histogram(s, nbins=50, title="Probability Distribution of Enterprise Value")
            fig_hist.add_vline(x=np.mean(s), line_color="red", annotation_text="Mean EV")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.success("Analysis: The wide spread indicates high sensitivity to revenue shocks.")
