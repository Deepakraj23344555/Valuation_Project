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
    /* Force Light Background and BLACK Text */
    .stTextInput>div>div>input { color: black !important; background-color: #F3E5AB !important; font-weight: bold; }
    .stNumberInput>div>div>input { color: black !important; background-color: #F3E5AB !important; font-weight: bold; }
    
    /* 10. SELECTBOX / DROPDOWN - HIGH CONTRAST FIX */
    /* The box itself */
    div[data-baseweb="select"] > div {
        background-color: #F3E5AB !important;
        color: black !important;
        font-weight: bold;
    }
    /* The text inside the box */
    div[data-baseweb="select"] span {
        color: black !important; 
    }
    /* The SVG arrow icon */
    div[data-baseweb="select"] svg {
        fill: black !important;
    }
    /* The Dropdown Menu Options */
    ul[data-baseweb="menu"] {
        background-color: #F3E5AB !important;
    }
    li[data-baseweb="option"] {
        color: black !important;
    }
    
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

def fetch_and_project_financials(ticker, years_forecast, growth_rate, margin_target):
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials.T
        if fin.empty: return None
        last_row = fin.iloc[0]
        try: base_rev = last_row['Total Revenue']
        except: base_rev = last_row.get('Revenue', 1000)
        current_year = datetime.now().year
        projected_data = []
        curr_rev = base_rev
        for i in range(1, years_forecast + 1):
            curr_rev = curr_rev * (1 + growth_rate)
            projected_data.append({
                'Year': current_year + i,
                'Revenue': curr_rev,
                'EBITDA_Margin': margin_target, 
                'D_and_A': curr_rev * 0.04,
                'CapEx': curr_rev * 0.05,
                'Change_in_NWC': curr_rev * 0.01
            })
        return pd.DataFrame(projected_data)
    except Exception as e:
        st.error(f"Projection Error: {e}")
        return None

def create_detailed_report(company_name, mkt_data, df_forecast, ev, wacc, tgr, upside, comps_df):
    """
    Generates a Multi-Page HTML Report.
    """
    # 1. Formatting Helpers
    def fmt_curr(x): return f"${x:,.0f}" if isinstance(x, (int, float)) else x
    def fmt_pct(x): return f"{x:.2%}" if isinstance(x, (int, float)) else x
    
    # 2. Convert Dataframes to HTML
    forecast_html = df_forecast.to_html(classes='table', index=False, float_format=lambda x: f"{x:,.0f}" if x > 1 else f"{x:.2%}")
    comps_html = comps_df.to_html(classes='table', index=False) if not comps_df.empty else "<p>No comparable data selected.</p>"
    
    # 3. HTML Structure
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; }}
            .container {{ width: 800px; margin: 0 auto; }}
            
            /* Colors */
            .primary {{ color: #8B4513; }}
            .secondary {{ color: #A0522D; }}
            .accent {{ color: #B8860B; }}
            
            /* Structure */
            .page {{ page-break-after: always; min-height: 900px; padding: 40px; border: 1px solid #eee; background: white; margin-bottom: 20px; }}
            .header {{ border-bottom: 4px solid #8B4513; padding-bottom: 10px; margin-bottom: 30px; }}
            .header h1 {{ margin: 0; font-size: 32px; text-transform: uppercase; }}
            .header .meta {{ color: #666; font-size: 14px; }}
            
            /* Tables */
            table.table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px; }}
            table.table th {{ background-color: #8B4513; color: white; padding: 8px; text-align: right; }}
            table.table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            table.table td:first-child {{ text-align: left; font-weight: bold; }}
            
            /* Metrics Grid */
            .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
            .card {{ background: #f8f9fa; padding: 15px; border-left: 5px solid #8B4513; }}
            .card .label {{ font-size: 12px; text-transform: uppercase; color: #666; }}
            .card .value {{ font-size: 24px; font-weight: bold; color: #333; }}
            
            h2 {{ color: #8B4513; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
            .footer {{ font-size: 10px; color: #999; text-align: center; margin-top: 50px; border-top: 1px solid #eee; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
        
            <div class="page">
                <div class="header">
                    <h1>Valuation Report: {company_name}</h1>
                    <div class="meta">Generated by GT Valuation Terminal | {datetime.now().strftime('%B %d, %Y')}</div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <div class="label">Enterprise Value (DCF)</div>
                        <div class="value">{fmt_curr(ev)}</div>
                    </div>
                    <div class="card">
                        <div class="label">WACC Applied</div>
                        <div class="value">{fmt_pct(wacc)}</div>
                    </div>
                    <div class="card">
                        <div class="label">Upside Case (95%)</div>
                        <div class="value">{fmt_curr(upside)}</div>
                    </div>
                </div>
                
                <h2>Company Profile</h2>
                <p><strong>Sector:</strong> Technology / General</p>
                <p>{mkt_data.get('summary', 'Company description not available.')[:500]}...</p>
                
                <h2>Market Snapshot</h2>
                <table class="table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Current Market Cap</td><td>{fmt_curr(mkt_data.get('mkt_cap', 0))}</td></tr>
                    <tr><td>Current Price</td><td>{fmt_curr(mkt_data.get('price', 0))}</td></tr>
                    <tr><td>52 Week High</td><td>{fmt_curr(mkt_data.get('high_52', 0))}</td></tr>
                    <tr><td>Risk Free Rate</td><td>{fmt_pct(mkt_data.get('rf_rate', 0))}</td></tr>
                </table>
                
                <div class="footer">CONFIDENTIAL - INTERNAL USE ONLY</div>
            </div>
            
            <div class="page">
                <div class="header">
                    <h1>Financial Forecast</h1>
                </div>
                
                <h2>Projected Performance (5 Years)</h2>
                <p>The following table outlines the pro-forma financial statement projected for the valuation period.</p>
                {forecast_html}
                
                <h2>Assumptions</h2>
                <ul>
                    <li><strong>Terminal Growth Rate:</strong> {fmt_pct(tgr)}</li>
                    <li><strong>Discount Rate (WACC):</strong> {fmt_pct(wacc)}</li>
                    <li><strong>Forecast Period:</strong> 5 Years</li>
                </ul>
                 <div class="footer">CONFIDENTIAL - INTERNAL USE ONLY</div>
            </div>

            <div class="page">
                <div class="header">
                    <h1>Valuation Details</h1>
                </div>
                
                <h2>DCF Methodology</h2>
                <p>The intrinsic value was derived using a standard Discounted Cash Flow (DCF) approach. Free Cash Flows (FCF) were projected for 5 years and discounted back to present value using the calculated WACC.</p>
                
                <div class="card" style="margin-top: 20px;">
                    <div class="label">Implied Enterprise Value</div>
                    <div class="value">{fmt_curr(ev)}</div>
                </div>

                <h2>Comparable Companies Analysis</h2>
                <p>Relative valuation based on selected peer group multiples.</p>
                {comps_html}
                
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
    st.caption("Professional Edition v4.0")
    st.markdown("---")
    
    nav = st.radio("Navigation", 
        ["🗂️ Project Setup", 
         "📈 Live Market Terminal", 
         "💎 DCF & 3D Sensitivity", 
         "🌍 Comps Regression",
         "⚡ Risk & Reporting"])
    
    st.markdown("---")
    st.info("💡 **Analyst Note:** Use the 'Live Market Terminal' to fetch real-time data before valuing.")

# --- 4. MAIN APP LOGIC ---

# --------------------------
# TAB 1: PROJECT SETUP (FORECASTING)
# --------------------------
if nav == "🗂️ Project Setup":
    st.title("🗂️ Project Initialization")
    st.markdown("Initialize your model. Choose **Quick Forecast** (Auto-Build) or **Pro Forecast** (Excel Upload).")
    
    # Toggle between Upload and Auto-Gen
    setup_mode = st.radio("Select Forecasting Mode:", ["⚡ Quick Forecast (Auto-Generate)", "📂 Pro Forecast (Excel Upload)"], horizontal=True)

    if setup_mode == "📂 Pro Forecast (Excel Upload)":
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 1. Template")
            st.write("Download the template, fill in your detailed projections, and upload.")
            template = generate_template()
            st.download_button("💾 Download Excel Template", data=template, file_name="GT_Model_Template.xlsx")
        
        with col2:
            st.markdown("### 2. Data Upload")
            uploaded_file = st.file_uploader("Import Financial Model", type=['xlsx'])
            if uploaded_file:
                df = pd.read_excel(uploaded_file)
                st.session_state['data'] = df
                st.success("✅ Financials Loaded")

    else: # Quick Forecast Mode
        st.markdown("### ⚡ Quick Forecast Builder")
        c1, c2, c3 = st.columns(3)
        with c1:
            auto_ticker = st.text_input("Ticker Symbol", "AAPL").upper()
        with c2:
            est_growth = st.slider("Est. Revenue Growth (%)", 0, 50, 10) / 100
        with c3:
            est_margin = st.slider("Target EBITDA Margin (%)", 5, 60, 30) / 100
            
        if st.button("🚀 Build Forecast Model"):
            with st.spinner("Fetching historicals and projecting future..."):
                df = fetch_and_project_financials(auto_ticker, 5, est_growth, est_margin)
                
                if df is not None:
                    st.session_state['data'] = df
                    st.session_state['ticker_name'] = auto_ticker 
                    # Fetch market data immediately for the report
                    mkt = get_market_data(auto_ticker)
                    if mkt: st.session_state['ticker_data'] = mkt
                    
                    st.success(f"✅ Model Built for {auto_ticker}")
                else:
                    st.error("Failed to fetch data.")

    # Show Data Preview if Loaded
    if 'data' in st.session_state:
        df = st.session_state['data']
        st.markdown("---")
        st.markdown("### Model Preview")
        
        # Calculate Implied EBITDA for the preview chart if not explicit column
        plot_df = df.copy()
        if 'EBITDA' not in plot_df.columns and 'EBITDA_Margin' in plot_df.columns:
            plot_df['EBITDA'] = plot_df['Revenue'] * plot_df['EBITDA_Margin']
            
        fig_prev = px.bar(plot_df, x='Year', y=['Revenue', 'EBITDA', 'CapEx'], barmode='group', 
                          title="Financial Projections", color_discrete_sequence=['#FFD700', '#8B4513', '#A0522D'])
        fig_prev.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
        st.plotly_chart(fig_prev, use_container_width=True)

# --------------------------
# TAB 2: LIVE MARKET TERMINAL
# --------------------------
elif nav == "📈 Live Market Terminal":
    st.title("📈 Live Market Terminal")
    
    col_search, col_status = st.columns([3, 1])
    with col_search:
        ticker = st.text_input("Search Ticker (Bloomberg/Yahoo Code)", placeholder="e.g. AAPL, TSLA, RELIANCE.NS")
    
    if ticker:
        with st.spinner(f"Fetching live data for {ticker}..."):
            mkt_data = get_market_data(ticker)
            
            if mkt_data:
                st.session_state['ticker_data'] = mkt_data # Save for WACC
                st.session_state['ticker_name'] = mkt_data['name']
                
                # --- A. HEADER METRICS ---
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Current Price", f"{mkt_data['currency']} {mkt_data['price']:,.2f}")
                m2.metric("Market Cap", f"${mkt_data['mkt_cap']/1e9:,.1f}B")
                m3.metric("Beta (5Y)", f"{mkt_data['beta']:.2f}")
                m4.metric("P/E Ratio", f"{mkt_data['pe_ratio']:.1f}x")
                m5.metric("10Y Treasury (Rf)", f"{mkt_data['rf_rate']:.2%}")
                
                # --- B. PROFESSIONAL CANDLESTICK CHART ---
                st.markdown("### Price Action (1 Year)")
                hist_df = mkt_data['history']
                
                fig_candle = go.Figure(data=[go.Candlestick(x=hist_df.index,
                    open=hist_df['Open'], high=hist_df['High'],
                    low=hist_df['Low'], close=hist_df['Close'],
                    increasing_line_color='#00CC96', decreasing_line_color='#EF553B')])
                
                fig_candle.update_layout(
                    title=f"{mkt_data['name']} - Technical View",
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#E6D5B8"),
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                fig_candle.update_xaxes(gridcolor='#3E3226')
                fig_candle.update_yaxes(gridcolor='#3E3226')
                st.plotly_chart(fig_candle, use_container_width=True)
                
                # --- C. WACC CALCULATOR INTEGRATION ---
                with st.expander("⚖️ WACC Calculation Module (Expand to Configure)", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        rf = st.number_input("Risk Free (%)", value=mkt_data['rf_rate']*100) / 100
                        beta = st.number_input("Beta", value=mkt_data['beta'])
                        erp = st.number_input("Equity Risk Premium (%)", value=6.0) / 100
                    with c2:
                        cost_debt = st.number_input("Pre-Tax Cost of Debt (%)", value=5.5) / 100
                        tax_rate = st.number_input("Tax Rate (%)", value=25.0) / 100
                    with c3:
                        equity_w = st.slider("Equity Weight %", 0, 100, 80) / 100
                        
                    # Live WACC Calculation
                    ke = rf + beta * erp
                    kd_net = cost_debt * (1 - tax_rate)
                    wacc = (ke * equity_w) + (kd_net * (1 - equity_w))
                    
                    st.session_state['wacc'] = wacc
                    st.session_state['tax_rate'] = tax_rate
                    
                    st.metric("✅ Calculated WACC", f"{wacc:.2%}", delta="Used in Valuation")

            else:
                st.error("Ticker not found. Please try again.")

# --------------------------
# TAB 3: DCF & 3D SENSITIVITY
# --------------------------
elif nav == "💎 DCF & 3D Sensitivity":
    if 'data' not in st.session_state:
        st.warning("⚠️ Please initialize model in 'Project Setup' first (Upload Excel or Auto-Generate).")
    else:
        st.title("💎 Intrinsic Valuation")
        
        # Pull global variables
        df = st.session_state['data'].copy()
        wacc = st.session_state.get('wacc', 0.10)
        tax = st.session_state.get('tax_rate', 0.25)
        tgr = st.slider("Terminal Growth Rate (%)", 1.0, 5.0, 2.5, step=0.1) / 100
        st.session_state['tgr'] = tgr
        
        # --- DCF CORE ENGINE ---
        if 'EBITDA' not in df.columns:
            df['EBITDA'] = df['Revenue'] * df['EBITDA_Margin']
            
        df['NOPAT'] = (df['EBITDA'] - df['D_and_A']) * (1 - tax)
        df['UFCF'] = df['NOPAT'] + df['D_and_A'] - df['CapEx'] - df['Change_in_NWC']
        
        df['Period'] = range(1, len(df) + 1)
        df['PV'] = df['UFCF'] / ((1 + wacc) ** df['Period'])
        
        tv = (df['UFCF'].iloc[-1] * (1 + tgr)) / (wacc - tgr)
        pv_tv = tv / ((1 + wacc) ** len(df))
        ev = df['PV'].sum() + pv_tv
        st.session_state['ev'] = ev
        
        # --- OUTPUT METRICS ---
        k1, k2, k3 = st.columns(3)
        k1.metric("Enterprise Value", f"${ev:,.0f}")
        k2.metric("Terminal Value Contribution", f"{(pv_tv/ev)*100:.1f}%")
        k3.metric("WACC", f"{wacc:.2%}")
        
        st.divider()
        
        # --- ADVANCED VISUALS ---
        tab_v1, tab_v2 = st.tabs(["2D Waterfall", "3D Sensitivity Surface"])
        
        with tab_v1:
            fig_wf = go.Figure(go.Waterfall(
                x = ["Sum of FCFs (5Y)", "Terminal Value", "Enterprise Value"],
                y = [df['PV'].sum(), pv_tv, ev],
                connector = {"line":{"color":"#E6D5B8"}},
                text = [f"${df['PV'].sum():,.0f}", f"${pv_tv:,.0f}", f"${ev:,.0f}"],
                textposition = "outside"
            ))
            fig_wf.update_layout(title="Valuation Bridge", paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
            st.plotly_chart(fig_wf, use_container_width=True)
            
        with tab_v2:
            st.markdown("#### Interactive Sensitivity Analysis (Rotate the Cube)")
            # Create 3D Mesh Data
            wacc_range = np.linspace(wacc-0.02, wacc+0.02, 20)
            tgr_range = np.linspace(tgr-0.01, tgr+0.01, 20)
            X, Y = np.meshgrid(wacc_range, tgr_range)
            
            # Vectorized calculation for surface
            last_cf = df['UFCF'].iloc[-1]
            pv_explicit = df['PV'].sum() 
            # Recalculate PV_TV for each point
            TV_mesh = (last_cf * (1 + Y)) / (X - Y)
            PV_TV_mesh = TV_mesh / ((1 + X) ** len(df))
            Z = pv_explicit + PV_TV_mesh
            
            fig_3d = go.Figure(data=[go.Surface(z=Z, x=X*100, y=Y*100, colorscale='YlOrBr')])
            
            fig_3d.update_layout(
                title="EV Sensitivity to WACC & Growth",
                scene = dict(
                    xaxis_title='WACC (%)',
                    yaxis_title='Growth (%)',
                    zaxis_title='Enterprise Value ($)',
                    xaxis=dict(backgroundcolor="#1E1915", gridcolor="gray", showbackground=True),
                    yaxis=dict(backgroundcolor="#1E1915", gridcolor="gray", showbackground=True),
                    zaxis=dict(backgroundcolor="#1E1915", gridcolor="gray", showbackground=True),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E6D5B8"),
                height=600
            )
            st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------
# TAB 4: COMPS REGRESSION
# --------------------------
elif nav == "🌍 Comps Regression":
    st.title("🌍 Relative Valuation (Regression)")
    st.markdown("Analyze valuation anomalies using Peer Regression.")
    
    default_tickers = "GOOGL, MSFT, META, AMZN, NFLX, NVDA"
    comps_input = st.text_input("Competitor Tickers", default_tickers)
    
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
                        data_points.append({
                            "Ticker": t,
                            "Revenue": rev,
                            "EV": ev_peer,
                            "EV/Rev": ev_peer/rev
                        })
                except: pass
                
        if data_points:
            comp_df = pd.DataFrame(data_points)
            st.session_state['comps'] = comp_df
            
            # --- SCATTER PLOT WITH REGRESSION ---
            fig_scat = px.scatter(comp_df, x="Revenue", y="EV", text="Ticker", 
                                  trendline="ols", 
                                  title="Market Regression: Size vs. Valuation",
                                  color_discrete_sequence=['#FFD700'])
            
            fig_scat.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
            fig_scat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color="#E6D5B8")
            )
            fig_scat.update_xaxes(showgrid=True, gridcolor='#3E3226')
            fig_scat.update_yaxes(showgrid=True, gridcolor='#3E3226')
            
            st.plotly_chart(fig_scat, use_container_width=True)
            
            st.markdown("### Peer Data Table")
            st.dataframe(comp_df.style.format({"Revenue": "${:,.0f}", "EV": "${:,.0f}", "EV/Rev": "{:.2f}x"}), use_container_width=True)
        else:
            st.error("No valid data found for these tickers.")

# --------------------------
# TAB 5: RISK & REPORTING
# --------------------------
elif nav == "⚡ Risk & Reporting":
    if 'ev' not in st.session_state:
        st.warning("⚠️ Calculate Value in Tab 3 first.")
    else:
        st.title("⚡ Risk Simulation & Reporting")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Monte Carlo Simulation")
            vol = st.slider("Revenue Volatility (σ)", 5, 40, 15) / 100
            iterations = st.selectbox("Iterations", [1000, 5000, 10000])
            
            if st.button("▶️ Run Simulation"):
                base_ev = st.session_state['ev']
                sims = base_ev * (1 + np.random.normal(0, vol, iterations))
                
                # Convergence Plot
                running_mean = [np.mean(sims[:i]) for i in range(1, len(sims), 50)]
                
                fig_conv = px.line(y=running_mean, title="Convergence of Mean Value (Law of Large Numbers)", 
                                   labels={'x': 'Iterations (x50)', 'y': 'Mean EV'})
                fig_conv.update_traces(line_color='#FFD700')
                fig_conv.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
                st.plotly_chart(fig_conv, use_container_width=True)
                
                # Histogram
                fig_hist = px.histogram(sims, nbins=50, title="Probability Distribution", color_discrete_sequence=['#8B4513'])
                fig_hist.add_vline(x=np.mean(sims), line_color="#FFD700", annotation_text="Mean")
                fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.session_state['upside'] = np.percentile(sims, 95)
        
        with c2:
            st.markdown("#### Client Deliverable")
            st.info("Generates a Detailed Multi-Page Report.")
            
            if st.button("🖨️ Generate Detailed PDF Report"):
                ticker_n = st.session_state.get('ticker_name', 'Company')
                
                # Gather all data for the report
                mkt_d = st.session_state.get('ticker_data', {})
                df_f = st.session_state.get('data', pd.DataFrame())
                ev_val = st.session_state.get('ev', 0)
                wacc_val = st.session_state.get('wacc', 0)
                tgr_val = st.session_state.get('tgr', 0.025)
                upside_val = st.session_state.get('upside', ev_val * 1.2) # Default assumption if simulation not run
                comps_d = st.session_state.get('comps', pd.DataFrame())
                
                # Generate HTML
                html = create_detailed_report(ticker_n, mkt_d, df_f, ev_val, wacc_val, tgr_val, upside_val, comps_d)
                
                b64 = base64.b64encode(html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="{ticker_n}_Valuation_Report.html" style="text-decoration:none;">' \
                       f'<button style="background-color:#FFD700; color:#2C241B; border:none; padding:10px 20px; font-weight:bold; border-radius:5px; cursor:pointer;">' \
                       f'📥 DOWNLOAD DETAILED REPORT</button></a>'
                st.markdown(href, unsafe_allow_html=True)
