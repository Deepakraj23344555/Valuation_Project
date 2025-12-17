import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from io import BytesIO
import base64
from datetime import datetime

# --- 1. CONFIGURATION & STYLING (Dark Coffee Theme) ---
st.set_page_config(page_title="GT Valuation Terminal", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp { background-color: #2C241B; }
    
    /* 2. Sidebar Background */
    section[data-testid="stSidebar"] { background-color: #1E1915; }
    
    /* 3. Typography */
    h1, h2, h3, h4 { color: #E6D5B8 !important; font-family: 'Helvetica Neue', sans-serif; }
    p, div, label, span { color: #F3F4F6; }
    
    /* 4. Sidebar Text */
    section[data-testid="stSidebar"] * { color: #D1C4E9 !important; }
    
    /* 5. Metrics */
    div[data-testid="stMetricValue"] { color: #FFD700; font-size: 26px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #E6D5B8; font-size: 14px; }
    
    /* 6. Buttons */
    .stButton>button {
        background-color: #8B4513; color: white !important;
        border: 1px solid #A0522D; border-radius: 4px;
        font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton>button:hover { background-color: #A0522D; border-color: #FFD700; }
    
    /* 7. Input Fields */
    .stTextInput>div>div>input { color: black; background-color: #E6D5B8; }
    .stNumberInput>div>div>input { color: black; }
    
    /* 8. Tabs */
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
        hist = stock.history(period="1y") # Get 1 Year Price History
        
        # Get Risk Free Rate
        treasury = yf.Ticker("^TNX") 
        rf_rate = treasury.history(period="1d")['Close'].iloc[-1] / 100
        
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
            "currency": info.get('currency', 'USD')
        }
    except:
        return None

def create_html_report(company_name, ev, wacc, upside, peers_data):
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica', sans-serif; color: #333; }}
            .header {{ background-color: #2C241B; color: #FFD700; padding: 25px; text-align: center; border-bottom: 5px solid #8B4513; }}
            h1 {{ margin: 0; font-size: 28px; }}
            .metric-box {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .metric-val {{ font-size: 24px; font-weight: bold; color: #8B4513; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ background-color: #8B4513; color: white; padding: 10px; text-align: left; }}
            td {{ border: 1px solid #ddd; padding: 10px; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>VALUATION SUMMARY: {company_name.upper()}</h1>
            <p>Grant Thornton Live Project | Date: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        
        <h3>1. Intrinsic Valuation (DCF Output)</h3>
        <div class="metric-box">
            Implied Enterprise Value: <span class="metric-val">${ev:,.0f}</span><br>
            WACC Applied: <span class="metric-val">{wacc:.2%}</span><br>
            Upside Potential (95% Conf.): <span class="metric-val">${upside:,.0f}</span>
        </div>
        
        <h3>2. Market Comparables</h3>
        {peers_data.to_html(index=False)}
        
        <br><br>
        <hr>
        <small>Generated via GT Valuation Terminal. Confidential.</small>
    </body>
    </html>
    """
    return html

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/242px-Python-logo-notext.svg.png", width=40)
    st.title("GT Terminal")
    st.caption("Professional Edition v3.0")
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
# TAB 1: PROJECT SETUP
# --------------------------
if nav == "🗂️ Project Setup":
    st.title("🗂️ Project Initialization")
    st.markdown("Load your financial model to begin the valuation workflow.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 1. Template")
        st.write("Required structure for financial inputs.")
        template = generate_template()
        st.download_button("💾 Download Excel Template", data=template, file_name="GT_Model_Template.xlsx")
    
    with col2:
        st.markdown("### 2. Data Upload")
        uploaded_file = st.file_uploader("Import Financial Model", type=['xlsx'])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.session_state['data'] = df
            
            # Show a nice preview chart immediately
            st.success("✅ Financials Loaded")
            fig_prev = px.bar(df, x='Year', y=['Revenue', 'EBITDA', 'CapEx'], barmode='group', 
                              title="Historical & Projected Financials", color_discrete_sequence=['#FFD700', '#8B4513', '#A0522D'])
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
                    st.session_state['ticker_name'] = mkt_data['name']
                    
                    st.metric("✅ Calculated WACC", f"{wacc:.2%}", delta="Used in Valuation")

            else:
                st.error("Ticker not found. Please try again.")

# --------------------------
# TAB 3: DCF & 3D SENSITIVITY
# --------------------------
elif nav == "💎 DCF & 3D Sensitivity":
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data in 'Project Setup' first.")
    else:
        st.title("💎 Intrinsic Valuation")
        
        # Pull global variables
        df = st.session_state['data'].copy()
        wacc = st.session_state.get('wacc', 0.10)
        tax = st.session_state.get('tax_rate', 0.25)
        tgr = st.slider("Terminal Growth Rate (%)", 1.0, 5.0, 2.5, step=0.1) / 100
        
        # --- DCF CORE ENGINE ---
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
            # Z = EV for each pair of (WACC, TGR)
            # Simplified PV of Explicit part (constant for graph speed approx, or recalculate full)
            pv_explicit = df['PV'].sum() 
            # Recalculate PV_TV for each point
            TV_mesh = (last_cf * (1 + Y)) / (X - Y)
            PV_TV_mesh = TV_mesh / ((1 + X) ** len(df))
            Z = pv_explicit + PV_TV_mesh
            
            fig_3d = go.Figure(data=[go.Surface(z=Z, x=X*100, y=Y*100, colorscale='Gold')])
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
                                  trendline="ols", # Ordinary Least Squares Regression
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
                
                # Convergence Plot (New Feature)
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
            st.info("Generates a One-Pager Summary for the Investment Committee.")
            
            if st.button("🖨️ Generate HTML Tear Sheet"):
                ticker_n = st.session_state.get('ticker_name', 'Project Alpha')
                wacc_u = st.session_state.get('wacc', 0)
                upside_u = st.session_state.get('upside', 0)
                comps_d = st.session_state.get('comps', pd.DataFrame())
                
                html = create_html_report(ticker_n, st.session_state['ev'], wacc_u, upside_u, comps_d)
                b64 = base64.b64encode(html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="{ticker_n}_Valuation.html" style="text-decoration:none;">' \
                       f'<button style="background-color:#FFD700; color:#2C241B; border:none; padding:10px 20px; font-weight:bold; border-radius:5px; cursor:pointer;">' \
                       f'📥 DOWNLOAD REPORT</button></a>'
                st.markdown(href, unsafe_allow_html=True)
