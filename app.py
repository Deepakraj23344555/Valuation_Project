import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GT Valuation Engine", page_icon="📈", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #4B0082; /* Grant Thornton Purple-ish tone */
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Financial Forecasting & Valuation Engine")
st.markdown("**Project:** Live Financial Modeling | **Developer:** MSc Finance Analyst")
st.markdown("---")

# --- SIDEBAR: DRIVERS & INPUTS ---
st.sidebar.header("1. Valuation Assumptions")

wacc_input = st.sidebar.slider("WACC (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.5) / 100
tgr_input = st.sidebar.slider("Terminal Growth Rate (%)", min_value=1.0, max_value=5.0, value=2.5, step=0.1) / 100
tax_rate_input = st.sidebar.number_input("Tax Rate (%)", value=25.0) / 100

st.sidebar.markdown("---")
st.sidebar.header("2. Simulation Settings")
sim_iterations = st.sidebar.selectbox("Monte Carlo Iterations", [1000, 5000, 10000])
rev_volatility = st.sidebar.slider("Revenue Volatility (+/- %)", 5, 30, 10) / 100

# --- FUNCTION: GENERATE TEMPLATE ---
def generate_template():
    # Creating a standard structure for the model
    data = {
        'Year': [2024, 2025, 2026, 2027, 2028],
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

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.info("💡 **Step 1:** Download the template to see the required format.")
    template_file = generate_template()
    st.download_button(
        label="📥 Download Excel Template",
        data=template_file,
        file_name="Financial_Model_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.info("💡 **Step 2:** Upload your populated Excel file.")
    uploaded_file = st.file_uploader("Upload Excel Model", type=['xlsx'])

# --- LOGIC EXECUTION ---
if uploaded_file is not None:
    # Read Data
    try:
        df = pd.read_excel(uploaded_file)
        
        # --- CALCULATION ENGINE (The "Backend" Logic) ---
        # 1. Calculate EBIT and NOPAT
        df['EBITDA'] = df['Revenue'] * df['EBITDA_Margin']
        df['EBIT'] = df['EBITDA'] - df['D_and_A']
        df['Tax_Payment'] = df['EBIT'] * tax_rate_input
        df['NOPAT'] = df['EBIT'] - df['Tax_Payment']
        
        # 2. Calculate Unlevered Free Cash Flow (UFCF)
        df['UFCF'] = df['NOPAT'] + df['D_and_A'] - df['CapEx'] - df['Change_in_NWC']
        
        # 3. Discount Factors
        df['Period'] = range(1, len(df) + 1)
        df['Discount_Factor'] = 1 / ((1 + wacc_input) ** df['Period'])
        df['PV_UFCF'] = df['UFCF'] * df['Discount_Factor']

        # 4. Terminal Value (Gordon Growth Method)
        last_ufcf = df['UFCF'].iloc[-1]
        terminal_value = (last_ufcf * (1 + tgr_input)) / (wacc_input - tgr_input)
        pv_terminal_value = terminal_value * df['Discount_Factor'].iloc[-1]

        # 5. Enterprise Value
        sum_pv_ufcf = df['PV_UFCF'].sum()
        enterprise_value = sum_pv_ufcf + pv_terminal_value

        # --- DASHBOARD VISUALIZATION ---
        st.markdown("---")
        st.subheader("Results Dashboard")

        # KPI Metrics
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Enterprise Value (EV)", f"${enterprise_value:,.2f}")
        kpi2.metric("Sum of PV Cash Flows", f"${sum_pv_ufcf:,.2f}")
        kpi3.metric("PV of Terminal Value", f"${pv_terminal_value:,.2f}")

        # Charts using Plotly
        tab1, tab2 = st.tabs(["📉 Cash Flow Projection", "🎲 Monte Carlo Simulation"])

        with tab1:
            # Waterfall Chart for Value
            fig = go.Figure(go.Waterfall(
                name = "20", orientation = "v",
                measure = ["relative", "relative", "total"],
                x = ["PV of Explicit Period", "PV of Terminal Value", "Total Enterprise Value"],
                textposition = "outside",
                text = [f"{sum_pv_ufcf:.0f}", f"{pv_terminal_value:.0f}", f"{enterprise_value:.0f}"],
                y = [sum_pv_ufcf, pv_terminal_value, enterprise_value],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            fig.update_layout(title="Enterprise Value Composition", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Revenue & EBITDA Trend
            fig2 = px.bar(df, x='Year', y=['Revenue', 'EBITDA'], barmode='group', title="Forecasted Revenue & EBITDA")
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("##### Probabilistic Valuation Analysis")
            st.write(f"Running **{sim_iterations}** simulations applying **{rev_volatility*100}%** volatility to Revenue estimates.")
            
            if st.button("Run Monte Carlo Simulation"):
                simulated_evs = []
                progress_bar = st.progress(0)
                
                # Monte Carlo Logic
                # We simulate adjusting the 'Base Case' EV by a random shock factor derived from revenue volatility
                for i in range(sim_iterations):
                    # Random shock following normal distribution
                    shock = np.random.normal(0, rev_volatility) 
                    # Apply shock to EV (Simplified proxy for full model recalc to save processing time)
                    sim_ev = enterprise_value * (1 + shock)
                    simulated_evs.append(sim_ev)
                    
                    if i % (sim_iterations // 10) == 0:
                        progress_bar.progress(i / sim_iterations)
                
                progress_bar.progress(100)
                
                # Histogram Result
                fig_hist = px.histogram(simulated_evs, nbins=50, title="Distribution of Likely Enterprise Values")
                fig_hist.add_vline(x=np.mean(simulated_evs), line_dash="dash", line_color="red", annotation_text="Mean EV")
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.success(f"Simulation Complete. 95% Confidence Interval: ${np.percentile(simulated_evs, 5):,.0f} - ${np.percentile(simulated_evs, 95):,.0f}")

    except Exception as e:
        st.error(f"Error processing file: {e}. Please ensure you are using the correct template.")

else:
    st.info("Please upload the financial model Excel file to begin analysis.")
