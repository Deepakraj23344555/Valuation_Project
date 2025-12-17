import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from io import BytesIO
import base64
from datetime import datetime

# --- 1. CONFIGURATION & STYLING (High Contrast Coffee Theme) ---
st.set_page_config(page_title="GT Valuation Terminal", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    /* 1. Main Background - Dark Coffee */
    .stApp { background-color: #2C241B; }
    
    /* 2. Sidebar Background - Darker Roast */
    section[data-testid="stSidebar"] { background-color: #1E1915; }
    
    /* 3. Typography - Headings (Latte Color) */
    h1, h2, h3, h4 { color: #E6D5B8 !important; font-family: 'Helvetica Neue', sans-serif; }
    
    /* 4. General Text - readable grey/white */
    p, div, span { color: #F3F4F6; }
    
    /* 5. LABELS - Force BRIGHT WHITE for visibility */
    label, .stMarkdown p { color: #FFFFFF !important; font-weight: 500; }
    
    /* 6. Sidebar Text - Light Purple/Grey */
    section[data-testid="stSidebar"] * { color: #D1C4E9 !important; }
    
    /* 7. METRICS */
    div[data-testid="stMetricValue"] { color: #FFD700; font-size: 26px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #E6D5B8; font-size: 14px; }
    
    /* 8. BUTTONS - Caramel/Bronze */
    .stButton>button {
        background-color: #8B4513; color: white !important;
        border: 1px solid #A0522D; border-radius: 4px;
        font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton>button:hover { background-color: #A0522D; border-color: #FFD700; }
    
    /* 9. INPUT FIELDS (Text, Number, Date) - HIGH CONTRAST FIX */
    .stTextInput>div>div>input { color: black !important; background-color: #F3E5AB !important; font-weight: bold; }
    .stNumberInput>div>div>input { color: black !important; background-color: #F3E5AB !important; font-weight: bold; }
    
    /* 10. SELECTBOX / DROPDOWN - HIGH CONTRAST FIX */
    div[data-baseweb="select"] > div { background-color: #F3E5AB !important; color: black !important; font-weight: bold; }
    div[data-baseweb="select"] span { color: black !important; }
    div[data-baseweb="select"] svg { fill: black !important; }
    ul[data-baseweb="menu"] { background-color: #F3E5AB !important; }
    li[data-baseweb="option"] { color: black !important; }
    
    /* 11. Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #3E3226; border-radius: 4px 4px 0 0; color: #E6D5B8; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #8B4513; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---

def generate_template():
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

@st.cache_data
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y") 
        try:
            rf_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        except:
            rf_rate = 0.045 

        return {
            "beta": info.get('beta', 1.0),
            "price": info.get('currentPrice', 0),
            "mkt_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "eps": info.get('trailingEps', 0),
            "high_52": info.get('fiftyTwoWeekHigh', 0),
            "low_52": info.get('fiftyTwoWeekLow', 0),
            "rf_rate": rf_rate,
            "name": info.get('longName', ticker),
            "history": hist,
            "currency": info.get('currency', 'USD'),
            "summary": info.get('longBusinessSummary', "No description available.")
        }
    except:
        return None

def project_scenario(base_rev, years_forecast, growth_rate, margin_target, current_year):
    """Helper to generate a single dataframe for one scenario"""
    data = []
    curr_rev = base_rev
    for i in range(1, years_forecast + 1):
        curr_rev = curr_rev * (1 + growth_rate)
        data.append({
            'Year': current_year + i,
            'Revenue': curr_rev,
            'EBITDA_Margin': margin_target, 
            'D_and_A': curr_rev * 0.04,
            'CapEx': curr_rev * 0.05,
            'Change_in_NWC': curr_rev * 0.01
        })
    return pd.DataFrame(data)

def fetch_and_project_scenarios(ticker, years_forecast, scenarios):
    """
    Generates 3 DataFrames: Bear, Base, Bull
    scenarios = {'Bear': (g, m), 'Base': (g, m), 'Bull': (g, m)}
    """
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials.T
        if fin.empty: return None

        last_row = fin.iloc[0]
        try: base_rev = last_row['Total Revenue']
        except: base_rev = last_row.get('Revenue', 1000)
        current_year = datetime.now().year
        
        results = {}
        for case_name, (growth, margin) in scenarios.items():
            results[case_name] = project_scenario(base_rev, years_forecast, growth, margin, current_year)
            
        return results
        
    except Exception as e:
        st.error(f"Projection Error: {e}")
        return None

def create_detailed_report(company_name, mkt_data, ev_dict, wacc, tgr, comps_df):
    """
    Generates Multi-Page HTML Report with Scenario Analysis
    """
    def fmt_curr(x): return f"${x:,.0f}" if isinstance(x, (int, float)) else x
    def fmt_pct(x): return f"{x:.2%}" if isinstance(x, (int, float)) else x
    
    comps_html = comps_df.to_html(classes='table', index=False) if not comps_df.empty else "<p>No comparable data selected.</p>"
    
    # Calculate Range
    ev_low = ev_dict.get('Bear', 0)
    ev_mid = ev_dict.get('Base', 0)
    ev_high = ev_dict.get('Bull', 0)
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; }}
            .container {{ width: 800px; margin: 0 auto; }}
            .page {{ page-break-after: always; min-height: 900px; padding: 40px; border: 1px solid #eee; background: white; margin-bottom: 20px; }}
            .header {{ border-bottom: 4px solid #8B4513; padding-bottom: 10px; margin-bottom: 30px; }}
            .header h1 {{ margin: 0; font-size: 32px; text-transform: uppercase; color: #8B4513; }}
            table.table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px; }}
            table.table th {{ background-color: #8B4513; color: white; padding: 8px; text-align: right; }}
            table.table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
            .card {{ background: #f8f9fa; padding: 15px; border-left: 5px solid #8B4513; }}
            .card .label {{ font-size: 12px; text-transform: uppercase; color: #666; }}
            .card .value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .footer {{ font-size: 10px; color: #999; text-align: center; margin-top: 50px; border-top: 1px solid #eee; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="page">
                <div class="header">
                    <h1>Valuation Report: {company_name}</h1>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <div class="label">Base Case EV</div>
                        <div class="value">{fmt_curr(ev_mid)}</div>
                    </div>
                    <div class="card">
                        <div class="label">WACC Applied</div>
                        <div class="value">{fmt_pct(wacc)}</div>
                    </div>
                    <div class="card">
                        <div class="label">Valuation Range</div>
                        <div class="value">{fmt_curr(ev_low)} - {fmt_curr(ev_high)}</div>
                    </div>
                </div>
                
                <h2>Scenario Analysis</h2>
                <p>Intrinsic value calculated under three distinct operating scenarios:</p>
                <table class="table">
                    <tr><th>Scenario</th><th>Enterprise Value</th></tr>
                    <tr><td style="text-align:left;">Bear Case (Low Growth)</td><td>{fmt_curr(ev_low)}</td></tr>
                    <tr><td style="text-align:left;">Base Case (Consensus)</td><td>{fmt_curr(ev_mid)}</td></tr>
                    <tr><td style="text-align:left;">Bull Case (High Growth)</td><td>{fmt_curr(ev_high)}</td></tr>
                </table>
                
                <h2>Company Profile</h2>
                <p>{mkt_data.get('summary', 'Company description not available.')[:500]}...</p>
                
                <div class="footer">CONFIDENTIAL - INTERNAL USE ONLY</div>
            </div>
            
            <div class="page">
                <div class="header"><h1>Market Analysis</h1></div>
                <h2>Comparable Companies</h2>
                {comps_html}
                
                <h2>Market Data</h2>
                <ul>
                    <li>Current Price: {fmt_curr(mkt_data.get('price', 0))}</li>
                    <li>Risk Free Rate: {fmt_pct(mkt_data.get('rf_rate', 0))}</li>
                </ul>
                <div class="footer">CONFIDENTIAL - INTERNAL USE ONLY</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/242px-Python-logo-notext.svg.png", width=40)
    st.title("GT Terminal")
    st.caption("Professional Edition v5.0")
    st.markdown("---")
    
    nav = st.radio("Navigation", 
        ["🗂️ Project Setup", 
         "📈 Live Market Terminal", 
         "💎 DCF & Scenario Analysis", 
         "🌍 Comps Regression",
         "⚡ Risk & Reporting"])
    
    st.markdown("---")
    st.info("💡 **Analyst Note:** Use the 'Live Market Terminal' to fetch real-time data before valuing.")

# --- 4. MAIN APP LOGIC ---

# --------------------------
# TAB 1: PROJECT SETUP (SCENARIO FORECASTING)
# --------------------------
if nav == "🗂️ Project Setup":
    st.title("🗂️ Project Initialization")
    st.markdown("Initialize your model. Use **Quick Forecast** to build Bear, Base, and Bull cases automatically.")
    
    setup_mode = st.radio("Select Forecasting Mode:", ["⚡ Quick Forecast (Scenario Builder)", "📂 Pro Forecast (Excel Upload)"], horizontal=True)

    if setup_mode == "📂 Pro Forecast (Excel Upload)":
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 1. Template")
            st.write("For detailed custom modeling (Base Case Only).")
            template = generate_template()
            st.download_button("💾 Download Excel Template", data=template, file_name="GT_Model_Template.xlsx")
        
        with col2:
            st.markdown("### 2. Data Upload")
            uploaded_file = st.file_uploader("Import Financial Model", type=['xlsx'])
            if uploaded_file:
                df = pd.read_excel(uploaded_file)
                # For excel upload, we just map it to 'Base' scenario
                st.session_state['scenarios'] = {'Base': df}
                st.session_state['mode'] = 'Excel'
                st.success("✅ Financials Loaded (Base Case)")

    else: # Quick Forecast Mode (SCENARIOS)
        st.markdown("### ⚡ Scenario Builder")
        ticker_col, _ = st.columns([1, 2])
        with ticker_col:
            auto_ticker = st.text_input("Ticker Symbol", "AAPL").upper()
            
        st.markdown("#### Define Scenarios")
        c1, c2, c3 = st.columns(3)
        
        # BEAR CASE
        with c1:
            st.markdown("🔴 **Bear Case**")
            bear_g = st.number_input("Growth % (Bear)", value=2.0, step=0.5) / 100
            bear_m = st.number_input("Margin % (Bear)", value=20.0, step=0.5) / 100
            
        # BASE CASE
        with c2:
            st.markdown("🟡 **Base Case**")
            base_g = st.number_input("Growth % (Base)", value=5.0, step=0.5) / 100
            base_m = st.number_input("Margin % (Base)", value=30.0, step=0.5) / 100
            
        # BULL CASE
        with c3:
            st.markdown("🟢 **Bull Case**")
            bull_g = st.number_input("Growth % (Bull)", value=10.0, step=0.5) / 100
            bull_m = st.number_input("Margin % (Bull)", value=35.0, step=0.5) / 100
            
        if st.button("🚀 Build Scenario Models"):
            with st.spinner("Projecting 3 distinct futures..."):
                scenarios_input = {
                    'Bear': (bear_g, bear_m),
                    'Base': (base_g, base_m),
                    'Bull': (bull_g, bull_m)
                }
                
                scenario_dfs = fetch_and_project_scenarios(auto_ticker, 5, scenarios_input)
                
                if scenario_dfs is not None:
                    st.session_state['scenarios'] = scenario_dfs
                    st.session_state['ticker_name'] = auto_ticker 
                    st.session_state['mode'] = 'Auto'
                    
                    # Fetch market data immediately for report
                    mkt = get_market_data(auto_ticker)
                    if mkt: st.session_state['ticker_data'] = mkt
                    
                    st.success(f"✅ Models Built: Bear, Base, Bull for {auto_ticker}")
                else:
                    st.error("Failed to fetch data.")

    # Show Data Preview
    if 'scenarios' in st.session_state:
        st.markdown("---")
        st.markdown("### Scenario Preview (Revenue)")
        
        # Combine revenues for plotting
        dfs = st.session_state['scenarios']
        combined = pd.DataFrame()
        for name, df in dfs.items():
            temp = df[['Year', 'Revenue']].copy()
            temp['Scenario'] = name
            combined = pd.concat([combined, temp])
            
        fig_prev = px.bar(combined, x='Year', y='Revenue', color='Scenario', barmode='group',
                          color_discrete_map={'Bear': '#E74C3C', 'Base': '#F1C40F', 'Bull': '#27AE60'})
        fig_prev.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
        st.plotly_chart(fig_prev, use_container_width=True)

# --------------------------
# TAB 2: LIVE MARKET TERMINAL
# --------------------------
elif nav == "📈 Live Market Terminal":
    st.title("📈 Live Market Terminal")
    col_search, col_status = st.columns([3, 1])
    with col_search:
        ticker = st.text_input("Search Ticker", placeholder="e.g. AAPL")
    
    if ticker:
        with st.spinner(f"Fetching live data for {ticker}..."):
            mkt_data = get_market_data(ticker)
            if mkt_data:
                st.session_state['ticker_data'] = mkt_data 
                st.session_state['ticker_name'] = mkt_data['name']
                
                # METRICS
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Current Price", f"{mkt_data['currency']} {mkt_data['price']:,.2f}")
                m2.metric("Market Cap", f"${mkt_data['mkt_cap']/1e9:,.1f}B")
                m3.metric("Beta", f"{mkt_data['beta']:.2f}")
                m4.metric("P/E", f"{mkt_data['pe_ratio']:.1f}x")
                m5.metric("Rf Rate", f"{mkt_data['rf_rate']:.2%}")
                
                # CHART
                hist_df = mkt_data['history']
                fig_candle = go.Figure(data=[go.Candlestick(x=hist_df.index,
                    open=hist_df['Open'], high=hist_df['High'],
                    low=hist_df['Low'], close=hist_df['Close'],
                    increasing_line_color='#00CC96', decreasing_line_color='#EF553B')])
                fig_candle.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
                fig_candle.update_xaxes(gridcolor='#3E3226')
                fig_candle.update_yaxes(gridcolor='#3E3226')
                st.plotly_chart(fig_candle, use_container_width=True)
                
                # WACC
                with st.expander("⚖️ WACC Calculation", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        rf = st.number_input("Risk Free", value=mkt_data['rf_rate']*100) / 100
                        beta = st.number_input("Beta", value=mkt_data['beta'])
                        erp = st.number_input("ERP", value=6.0) / 100
                    with c2:
                        cost_debt = st.number_input("Cost of Debt", value=5.5) / 100
                        tax_rate = st.number_input("Tax Rate", value=25.0) / 100
                    with c3:
                        equity_w = st.slider("Equity Weight", 0, 100, 80) / 100
                        
                    wacc = ((rf + beta * erp) * equity_w) + ((cost_debt * (1 - tax_rate)) * (1 - equity_w))
                    st.session_state['wacc'] = wacc
                    st.session_state['tax_rate'] = tax_rate
                    st.metric("✅ WACC", f"{wacc:.2%}")

# --------------------------
# TAB 3: DCF & SCENARIO ANALYSIS
# --------------------------
elif nav == "💎 DCF & Scenario Analysis":
    if 'scenarios' not in st.session_state:
        st.warning("⚠️ Please build models in 'Project Setup' first.")
    else:
        st.title("💎 Intrinsic Valuation: Scenario Analysis")
        
        # Globals
        scenarios = st.session_state['scenarios']
        wacc = st.session_state.get('wacc', 0.10)
        tax = st.session_state.get('tax_rate', 0.25)
        tgr = st.slider("Terminal Growth Rate (%)", 1.0, 5.0, 2.5, step=0.1) / 100
        st.session_state['tgr'] = tgr
        
        ev_results = {}
        
        # --- CALCULATE EV FOR ALL SCENARIOS ---
        for name, df in scenarios.items():
            df_calc = df.copy()
            # Calculate FCF if missing
            if 'EBITDA' not in df_calc.columns:
                df_calc['EBITDA'] = df_calc['Revenue'] * df_calc['EBITDA_Margin']
            
            df_calc['NOPAT'] = (df_calc['EBITDA'] - df_calc['D_and_A']) * (1 - tax)
            df_calc['UFCF'] = df_calc['NOPAT'] + df_calc['D_and_A'] - df_calc['CapEx'] - df_calc['Change_in_NWC']
            
            # Discount
            df_calc['Period'] = range(1, len(df_calc) + 1)
            df_calc['PV'] = df_calc['UFCF'] / ((1 + wacc) ** df_calc['Period'])
            
            # Terminal Value
            tv = (df_calc['UFCF'].iloc[-1] * (1 + tgr)) / (wacc - tgr)
            pv_tv = tv / ((1 + wacc) ** len(df_calc))
            
            ev_results[name] = df_calc['PV'].sum() + pv_tv
            
            # Save Base Case Detailed DF for 3D chart
            if name == 'Base':
                st.session_state['base_df'] = df_calc

        st.session_state['ev_results'] = ev_results # Save for reporting
        
        # --- DISPLAY RESULTS ---
        c1, c2, c3 = st.columns(3)
        c1.metric("🐻 Bear Case EV", f"${ev_results.get('Bear', 0):,.0f}")
        c2.metric("🟡 Base Case EV", f"${ev_results.get('Base', 0):,.0f}")
        c3.metric("🟢 Bull Case EV", f"${ev_results.get('Bull', 0):,.0f}")
        
        st.divider()
        
        # --- FOOTBALL FIELD CHART ---
        st.markdown("#### Valuation Range (Football Field)")
        
        fig_range = go.Figure()
        
        colors = {'Bear': '#E74C3C', 'Base': '#F1C40F', 'Bull': '#27AE60'}
        
        for name, val in ev_results.items():
            fig_range.add_trace(go.Bar(
                x=[val], y=[name], orientation='h', 
                marker_color=colors.get(name, 'grey'),
                text=f"${val:,.0f}", textposition='auto',
                name=name
            ))
            
        fig_range.update_layout(
            title="Enterprise Value by Scenario",
            xaxis_title="Enterprise Value ($)",
            yaxis_title="Scenario",
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"),
            height=300
        )
        st.plotly_chart(fig_range, use_container_width=True)

        # --- 3D SENSITIVITY (ON BASE CASE) ---
        st.divider()
        st.markdown("#### 3D Sensitivity (Base Case)")
        
        df_base = st.session_state.get('base_df')
        if df_base is not None:
            wacc_range = np.linspace(wacc-0.02, wacc+0.02, 20)
            tgr_range = np.linspace(tgr-0.01, tgr+0.01, 20)
            X, Y = np.meshgrid(wacc_range, tgr_range)
            
            last_cf = df_base['UFCF'].iloc[-1]
            pv_explicit = df_base['PV'].sum()
            TV_mesh = (last_cf * (1 + Y)) / (X - Y)
            PV_TV_mesh = TV_mesh / ((1 + X) ** len(df_base))
            Z = pv_explicit + PV_TV_mesh
            
            fig_3d = go.Figure(data=[go.Surface(z=Z, x=X*100, y=Y*100, colorscale='YlOrBr')])
            fig_3d.update_layout(
                scene = dict(xaxis_title='WACC', yaxis_title='Growth', zaxis_title='EV'),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"), height=500
            )
            st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------
# TAB 4: COMPS REGRESSION
# --------------------------
elif nav == "🌍 Comps Regression":
    st.title("🌍 Relative Valuation")
    comps_input = st.text_input("Competitor Tickers", "GOOGL, MSFT, META, AMZN, NFLX")
    
    if st.button("🔄 Analyze Peers"):
        tickers = [x.strip() for x in comps_input.split(',')]
        data_points = []
        with st.spinner("Fetching Peer Data..."):
            for t in tickers:
                try:
                    i = yf.Ticker(t).info
                    rev = i.get('totalRevenue', 0)
                    ev_peer = i.get('enterpriseValue', 0)
                    if rev > 0 and ev_peer > 0:
                        data_points.append({"Ticker": t, "Revenue": rev, "EV": ev_peer, "EV/Rev": ev_peer/rev})
                except: pass
                
        if data_points:
            comp_df = pd.DataFrame(data_points)
            st.session_state['comps'] = comp_df
            fig_scat = px.scatter(comp_df, x="Revenue", y="EV", text="Ticker", trendline="ols",
                                  title="Market Regression", color_discrete_sequence=['#FFD700'])
            fig_scat.update_traces(textposition='top center', marker=dict(size=12))
            fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
            st.plotly_chart(fig_scat, use_container_width=True)
            st.dataframe(comp_df.style.format({"Revenue": "${:,.0f}", "EV": "${:,.0f}", "EV/Rev": "{:.2f}x"}))

# --------------------------
# TAB 5: RISK & REPORTING
# --------------------------
elif nav == "⚡ Risk & Reporting":
    if 'ev_results' not in st.session_state:
        st.warning("⚠️ Calculate Value in Tab 3 first.")
    else:
        st.title("⚡ Risk & Reporting")
        
        ev_results = st.session_state['ev_results']
        base_ev = ev_results.get('Base', 0)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Monte Carlo (Base Case)")
            vol = st.slider("Revenue Volatility (σ)", 5, 40, 15) / 100
            if st.button("▶️ Run Simulation"):
                sims = base_ev * (1 + np.random.normal(0, vol, 1000))
                fig_hist = px.histogram(sims, title="Probability Distribution", color_discrete_sequence=['#8B4513'])
                fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
                st.plotly_chart(fig_hist, use_container_width=True)
                
        with c2:
            st.markdown("#### Scenario Report")
            st.info("Generates a report comparing Bear, Base, and Bull cases.")
            
            if st.button("🖨️ Generate PDF Report"):
                ticker_n = st.session_state.get('ticker_name', 'Company')
                mkt_d = st.session_state.get('ticker_data', {})
                wacc_u = st.session_state.get('wacc', 0)
                tgr_u = st.session_state.get('tgr', 0.025)
                comps_d = st.session_state.get('comps', pd.DataFrame())
                
                html = create_detailed_report(ticker_n, mkt_d, ev_results, wacc_u, tgr_u, comps_d)
                b64 = base64.b64encode(html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="{ticker_n}_Valuation_Report.html" style="text-decoration:none;">' \
                       f'<button style="background-color:#FFD700; color:#2C241B; border:none; padding:10px 20px;">📥 DOWNLOAD REPORT</button></a>'
                st.markdown(href, unsafe_allow_html=True)
