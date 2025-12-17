import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from io import BytesIO
import base64
from datetime import datetime
from textblob import TextBlob

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="GT Valuation Terminal", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #2C241B; }
    section[data-testid="stSidebar"] { background-color: #1E1915; }
    h1, h2, h3, h4 { color: #E6D5B8 !important; font-family: 'Helvetica Neue', sans-serif; }
    p, div, span, li { color: #F3F4F6; }
    label, .stMarkdown p { color: #FFFFFF !important; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #FFD700; font-size: 26px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #E6D5B8; font-size: 14px; }
    .stButton>button {
        background-color: #8B4513; color: white !important;
        border: 1px solid #A0522D; border-radius: 4px;
        font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton>button:hover { background-color: #A0522D; border-color: #FFD700; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { 
        color: black !important; background-color: #F3E5AB !important; font-weight: bold; 
    }
    div[data-baseweb="select"] > div { background-color: #F3E5AB !important; color: black !important; font-weight: bold; }
    div[data-baseweb="select"] span { color: black !important; }
    ul[data-baseweb="menu"] { background-color: #F3E5AB !important; }
    li[data-baseweb="option"] { color: black !important; }
    a { color: #FFD700 !important; text-decoration: none; }
    [data-testid="stDataFrame"] { border: 1px solid #8B4513; }
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
        hist = stock.history(period="2y")
        try:
            rf_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        except:
            rf_rate = 0.045

        # Balance Sheet Safe Extraction
        bs = stock.balance_sheet
        def get_bs_value(key):
            if key in bs.index: return bs.loc[key].iloc[0]
            return 0
            
        total_assets = get_bs_value('Total Assets')
        total_liab = get_bs_value('Total Liabilities Net Minority Interest')
        retained_earnings = get_bs_value('Retained Earnings')
        
        # Working Capital Calculation
        curr_assets = get_bs_value('Current Assets')
        curr_liab = get_bs_value('Current Liabilities')
        working_capital = curr_assets - curr_liab
        
        total_debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        shares = info.get('sharesOutstanding', 1)
        
        inc = stock.financials
        ebit = inc.loc['EBIT'].iloc[0] if 'EBIT' in inc.index else 0
        revenue = inc.loc['Total Revenue'].iloc[0] if 'Total Revenue' in inc.index else 0
        
        sector = info.get('sector', 'Unknown')

        return {
            "beta": info.get('beta', 1.0),
            "price": info.get('currentPrice', 0),
            "mkt_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "high_52": info.get('fiftyTwoWeekHigh', 0),
            "rf_rate": rf_rate,
            "name": info.get('longName', ticker),
            "history": hist,
            "currency": info.get('currency', 'USD'),
            "summary": info.get('longBusinessSummary', "No description available."),
            "sector": sector,
            "total_assets": total_assets,
            "total_liab": total_liab,
            "retained_earnings": retained_earnings,
            "working_capital": working_capital,
            "ebit": ebit,
            "revenue": revenue,
            "debt": total_debt,
            "cash": cash,
            "shares": shares
        }
    except:
        return None

def get_historical_profitability(ticker):
    try:
        stock = yf.Ticker(ticker)
        fin = stock.financials.T
        if fin.empty: return None
        
        df = pd.DataFrame(index=fin.index)
        
        # Revenue Check
        if 'Total Revenue' in fin.columns:
            df['Revenue'] = fin['Total Revenue']
        elif 'Revenue' in fin.columns:
            df['Revenue'] = fin['Revenue']
        else:
            return None 

        # 1. Gross Margin (SKIP FOR BANKS if missing)
        if 'Gross Profit' in fin.columns:
            df['Gross_Margin'] = (fin['Gross Profit'] / df['Revenue']) * 100
        else:
            df['Gross_Margin'] = np.nan 

        # 2. Operating Margin
        op_col = None
        if 'Operating Income' in fin.columns: op_col = 'Operating Income'
        elif 'EBIT' in fin.columns: op_col = 'EBIT'
        elif 'Pretax Income' in fin.columns: op_col = 'Pretax Income' # Proxy for Banks
        
        if op_col:
             df['Operating_Margin'] = (fin[op_col] / df['Revenue']) * 100
        else:
             df['Operating_Margin'] = np.nan

        # 3. Net Margin
        if 'Net Income' in fin.columns:
            df['Net_Margin'] = (fin['Net Income'] / df['Revenue']) * 100
        else:
            df['Net_Margin'] = np.nan
        
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # Drop columns that are completely empty (fixes the Bank crash)
        df = df.dropna(axis=1, how='all')
        
        return df
    except:
        return None

def project_scenario(base_rev, years_forecast, growth_rate, margin_target, current_year):
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

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_detailed_report(company_name, mkt_data, ev_dict, wacc, comps_df, base_df):
    ev_mid = ev_dict.get('Base', 0)
    ev_low = ev_dict.get('Bear', 0)
    ev_high = ev_dict.get('Bull', 0)
    
    def fmt_curr(x): return f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"
    
    comps_html = comps_df.to_html(classes='table', index=False) if not comps_df.empty else "<p>No peers found.</p>"
    
    forecast_html = "<p>No forecast data.</p>"
    if base_df is not None:
        display_df = base_df[['Year', 'Revenue', 'EBITDA', 'UFCF']].copy()
        forecast_html = display_df.to_html(classes='table', index=False, float_format=lambda x: f"{x:,.0f}")

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, Arial, sans-serif; color: #333; }}
            .header {{ border-bottom: 4px solid #8B4513; padding-bottom: 10px; margin-bottom: 30px; }}
            h1 {{ color: #8B4513; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ background-color: #8B4513; color: white; padding: 8px; }}
            td {{ border: 1px solid #ddd; padding: 8px; }}
        </style>
    </head>
    <body>
        <div class="header"><h1>Valuation Report: {company_name}</h1></div>
        <h3>Base Case EV: {fmt_curr(ev_mid)} | WACC: {wacc:.2%}</h3>
        <p><strong>Sector:</strong> {mkt_data.get('sector', 'N/A')}</p>
        <p>{mkt_data.get('summary', '')[:600]}...</p>
        
        <h3>Valuation Range</h3>
        <ul>
            <li>Bear: {fmt_curr(ev_low)}</li>
            <li>Base: {fmt_curr(ev_mid)}</li>
            <li>Bull: {fmt_curr(ev_high)}</li>
        </ul>
        
        <h3>Financial Forecast (Base Case)</h3>
        {forecast_html}
        
        <h3>Peer Analysis</h3>
        {comps_html}
    </body>
    </html>
    """
    return html

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/242px-Python-logo-notext.svg.png", width=40)
    st.title("GT Terminal")
    st.caption("Ultimate Edition v10.0 (Bank Fix)")
    st.markdown("---")
    
    nav = st.radio("Navigation", 
        ["🗂️ Project Setup", 
         "📈 Live Market Terminal", 
         "💎 DCF & Scenario Analysis", 
         "🌍 Comps Regression",
         "⚡ Risk & Reporting",
         "🏥 Financial Health (Z-Score)",
         "📊 Deep Dive"])
    
    st.markdown("---")
    st.info("💡 **Analyst Note:** Use the 'Live Market Terminal' to fetch real-time data before valuing.")

# --- 4. MAIN APP LOGIC ---

# --------------------------
# TAB 1: PROJECT SETUP
# --------------------------
if nav == "🗂️ Project Setup":
    st.title("🗂️ Project Initialization")
    st.markdown("Initialize your model. Use **Quick Forecast** to build Scenarios.")
    
    setup_mode = st.radio("Select Forecasting Mode:", ["⚡ Quick Forecast (Scenario Builder)", "📂 Pro Forecast (Excel Upload)"], horizontal=True)

    if setup_mode == "📂 Pro Forecast (Excel Upload)":
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 1. Template")
            template = generate_template()
            st.download_button("💾 Download Excel Template", data=template, file_name="GT_Model_Template.xlsx")
        with col2:
            st.markdown("### 2. Data Upload")
            uploaded_file = st.file_uploader("Import Financial Model", type=['xlsx'])
            if uploaded_file:
                df = pd.read_excel(uploaded_file)
                st.session_state['scenarios'] = {'Base': df}
                st.session_state['mode'] = 'Excel'
                st.success("✅ Financials Loaded (Base Case)")

    else: 
        st.markdown("### ⚡ Scenario Builder")
        ticker_col, _ = st.columns([1, 2])
        with ticker_col:
            auto_ticker = st.text_input("Ticker Symbol", "AAPL").upper()
            
        st.markdown("#### Define Scenarios")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("🔴 **Bear Case**")
            bear_g = st.number_input("Growth % (Bear)", value=2.0) / 100
            bear_m = st.number_input("Margin % (Bear)", value=20.0) / 100
        with c2:
            st.markdown("🟡 **Base Case**")
            base_g = st.number_input("Growth % (Base)", value=5.0) / 100
            base_m = st.number_input("Margin % (Base)", value=30.0) / 100
        with c3:
            st.markdown("🟢 **Bull Case**")
            bull_g = st.number_input("Growth % (Bull)", value=10.0) / 100
            bull_m = st.number_input("Margin % (Bull)", value=35.0) / 100
            
        if st.button("🚀 Build Scenario Models"):
            with st.spinner("Projecting futures..."):
                scenarios_input = {'Bear': (bear_g, bear_m), 'Base': (base_g, base_m), 'Bull': (bull_g, bull_m)}
                scenario_dfs = fetch_and_project_scenarios(auto_ticker, 5, scenarios_input)
                
                if scenario_dfs:
                    st.session_state['scenarios'] = scenario_dfs
                    st.session_state['ticker_name'] = auto_ticker 
                    st.session_state['ticker_symbol'] = auto_ticker 
                    
                    mkt = get_market_data(auto_ticker)
                    if mkt: st.session_state['ticker_data'] = mkt
                    
                    # Save Base for 3D Plot
                    st.session_state['base_df'] = scenario_dfs['Base']
                    st.success(f"✅ Models Built for {auto_ticker}")
                else:
                    st.error("Failed to fetch data.")

    if 'scenarios' in st.session_state:
        st.markdown("---")
        st.markdown("### 📤 Export Models")
        combined_export = pd.DataFrame()
        for name, df in st.session_state['scenarios'].items():
            temp = df.copy()
            temp['Scenario'] = name
            combined_export = pd.concat([combined_export, temp])
        csv = combined_export.to_csv(index=False).encode('utf-8')
        st.download_button(label="💾 Download All Scenarios (CSV)", data=csv, file_name="Scenarios.csv", mime='text/csv')

# --------------------------
# TAB 2: LIVE MARKET
# --------------------------
elif nav == "📈 Live Market Terminal":
    st.title("📈 Live Market Terminal")
    col_search, col_status = st.columns([3, 1])
    with col_search:
        ticker = st.text_input("Search Ticker", placeholder="e.g. AAPL, HDFCBANK.NS")
    
    if ticker:
        with st.spinner(f"Fetching live data for {ticker}..."):
            mkt_data = get_market_data(ticker)
            if mkt_data:
                st.session_state['ticker_data'] = mkt_data 
                st.session_state['ticker_name'] = mkt_data['name']
                st.session_state['ticker_symbol'] = ticker
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Price", f"{mkt_data['currency']} {mkt_data['price']:,.2f}")
                m2.metric("Mkt Cap", f"${mkt_data['mkt_cap']/1e9:,.1f}B")
                m3.metric("Beta", f"{mkt_data['beta']:.2f}")
                m4.metric("P/E", f"{mkt_data['pe_ratio']:.1f}x")
                m5.metric("Rf Rate", f"{mkt_data['rf_rate']:.2%}")
                
                st.markdown("### Technical Analysis")
                hist_df = mkt_data['history'].copy()
                hist_df['SMA_50'] = hist_df['Close'].rolling(window=50).mean()
                hist_df['SMA_200'] = hist_df['Close'].rolling(window=200).mean()
                hist_df['RSI'] = calculate_rsi(hist_df['Close'])
                
                fig_candle = go.Figure()
                fig_candle.add_trace(go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
                    low=hist_df['Low'], close=hist_df['Close'], name='Price'))
                fig_candle.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_50'], line=dict(color='#FFD700'), name='SMA 50'))
                fig_candle.add_trace(go.Scatter(x=hist_df.index, y=hist_df['SMA_200'], line=dict(color='#00BFFF'), name='SMA 200'))
                fig_candle.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_candle, use_container_width=True)
                
                st.markdown("### 📰 News Sentiment")
                try:
                    news = yf.Ticker(ticker).news
                    if news:
                        for item in news[:3]:
                            title = item.get('title', 'No Title')
                            link = item.get('link', '#')
                            blob = TextBlob(title)
                            polarity = blob.sentiment.polarity
                            if polarity > 0.1: icon, color = "🟢 Positive", "green"
                            elif polarity < -0.1: icon, color = "🔴 Negative", "red"
                            else: icon, color = "⚪ Neutral", "grey"
                            st.markdown(f"**[{title}]({link})**")
                            st.caption(f"Sentiment: :{color}[{icon}]")
                            st.divider()
                    else: st.info("No news found.")
                except: st.info("News unavailable.")
                
                with st.expander("⚖️ WACC Calculation", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        rf = st.number_input("Risk Free", value=mkt_data['rf_rate']*100)
                        beta = st.number_input("Beta", value=mkt_data['beta'])
                        erp = st.number_input("ERP", value=0.06)
                    with c2:
                        cost_debt = st.number_input("Cost of Debt", value=0.055)
                        tax_rate = st.number_input("Tax Rate", value=0.25)
                    with c3:
                        equity_w = st.slider("Equity Weight", 0.0, 1.0, 0.8)
                    wacc = ((rf + beta * erp) * equity_w) + ((cost_debt * (1 - tax_rate)) * (1 - equity_w))
                    st.session_state['wacc'] = wacc
                    st.metric("✅ WACC", f"{wacc:.2%}")

# --------------------------
# TAB 3: DCF
# --------------------------
elif nav == "💎 DCF & Scenario Analysis":
    if 'scenarios' not in st.session_state:
        st.warning("⚠️ Please build models in 'Project Setup' first.")
    else:
        st.title("💎 Intrinsic Valuation")
        scenarios = st.session_state['scenarios']
        wacc = st.session_state.get('wacc', 0.10)
        tgr = st.slider("Terminal Growth Rate (%)", 1.0, 5.0, 2.5, step=0.1) / 100
        
        mkt = st.session_state.get('ticker_data', {})
        net_debt = mkt.get('debt', 0) - mkt.get('cash', 0)
        shares = mkt.get('shares', 1)
        
        ev_results = {}
        for name, df in scenarios.items():
            df_calc = df.copy()
            if 'EBITDA' not in df_calc.columns: df_calc['EBITDA'] = df_calc['Revenue'] * df_calc['EBITDA_Margin']
            df_calc['NOPAT'] = (df_calc['EBITDA'] - df_calc['D_and_A']) * (1 - 0.25)
            df_calc['UFCF'] = df_calc['NOPAT'] + df_calc['D_and_A'] - df_calc['CapEx'] - df_calc['Change_in_NWC']
            
            df_calc['Period'] = range(1, len(df_calc) + 1)
            df_calc['PV'] = df_calc['UFCF'] / ((1 + wacc) ** df_calc['Period'])
            tv = (df_calc['UFCF'].iloc[-1] * (1 + tgr)) / (wacc - tgr)
            pv_tv = tv / ((1 + wacc) ** len(df_calc))
            ev_results[name] = df_calc['PV'].sum() + pv_tv
            
            if name == 'Base': st.session_state['base_df'] = df_calc

        st.session_state['ev_results'] = ev_results
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🐻 Bear Case EV", f"${ev_results.get('Bear', 0):,.0f}")
        c2.metric("🟡 Base Case EV", f"${ev_results.get('Base', 0):,.0f}")
        c3.metric("🟢 Bull Case EV", f"${ev_results.get('Bull', 0):,.0f}")
        
        fig_range = go.Figure()
        colors = {'Bear': '#E74C3C', 'Base': '#F1C40F', 'Bull': '#27AE60'}
        for name, val in ev_results.items():
            fig_range.add_trace(go.Bar(x=[val], y=[name], orientation='h', marker_color=colors.get(name, 'grey'),
                                       text=f"${val:,.0f}", textposition='auto', name=name))
        fig_range.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
        st.plotly_chart(fig_range, use_container_width=True)

        st.divider()
        st.markdown("#### 🧊 3D Sensitivity Surface")
        
        base_df_sens = st.session_state.get('base_df')
        if base_df_sens is not None:
            wacc_range = np.linspace(wacc-0.02, wacc+0.02, 20)
            tgr_range = np.linspace(tgr-0.01, tgr+0.01, 20)
            X, Y = np.meshgrid(wacc_range, tgr_range)
            last_cf = base_df_sens['UFCF'].iloc[-1]
            pv_explicit = base_df_sens['PV'].sum()
            TV_mesh = (last_cf * (1 + Y)) / (X - Y)
            PV_TV_mesh = TV_mesh / ((1 + X) ** len(base_df_sens))
            Z = pv_explicit + PV_TV_mesh
            
            fig_3d = go.Figure(data=[go.Surface(z=Z, x=X*100, y=Y*100, colorscale='YlOrBr')])
            fig_3d.update_layout(scene=dict(xaxis_title='WACC', yaxis_title='Growth', zaxis_title='EV'),
                                 paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"), height=600)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("Base case data missing. Please rebuild model in Tab 1.")

# --------------------------
# TAB 4: COMPS
# --------------------------
elif nav == "🌍 Comps Regression":
    st.title("🌍 Relative Valuation")
    comps_input = st.text_input("Competitors", "GOOGL, MSFT, META, AMZN, NFLX")
    if st.button("🔄 Analyze Peers"):
        tickers = [x.strip() for x in comps_input.split(',')]
        data_points = []
        with st.spinner("Fetching Peer Data..."):
            for t in tickers:
                try:
                    i = yf.Ticker(t).info
                    rev = i.get('totalRevenue', 0)
                    ev_peer = i.get('enterpriseValue', 0)
                    if rev > 0:
                        data_points.append({
                            "Ticker": t, "Revenue": rev, "EV": ev_peer, "EV/Rev": ev_peer/rev,
                            "P/E": i.get('trailingPE', 0), "ROE": i.get('returnOnEquity', 0)
                        })
                except: pass
        if data_points:
            comp_df = pd.DataFrame(data_points)
            st.session_state['comps'] = comp_df
            fig_scat = px.scatter(comp_df, x="Revenue", y="EV", text="Ticker", trendline="ols",
                                  title="Market Regression", color_discrete_sequence=['#FFD700'])
            fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
            st.plotly_chart(fig_scat, use_container_width=True)
            st.dataframe(comp_df.style.format({"Revenue": "${:,.0f}", "EV": "${:,.0f}"}))

# --------------------------
# TAB 5: RISK
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
            st.markdown("#### Monte Carlo")
            vol = st.slider("Volatility (σ)", 5, 40, 15) / 100
            iterations = st.selectbox("Simulations", [1000, 5000, 10000])
            if st.button("▶️ Run Simulation"):
                sims = base_ev * (1 + np.random.normal(0, vol, iterations))
                fig_hist = px.histogram(sims, title="Distribution", color_discrete_sequence=['#8B4513'])
                fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
                st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            st.markdown("#### PDF Report")
            if st.button("🖨️ Generate Report"):
                ticker_n = st.session_state.get('ticker_name', 'Company')
                mkt_d = st.session_state.get('ticker_data', {})
                wacc_u = st.session_state.get('wacc', 0)
                comps_d = st.session_state.get('comps', pd.DataFrame())
                base_d = st.session_state.get('base_df')
                html = create_detailed_report(ticker_n, mkt_d, ev_results, wacc_u, comps_d, base_d)
                b64 = base64.b64encode(html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="{ticker_n}_Report.html"><button style="padding:10px;">📥 Download</button></a>'
                st.markdown(href, unsafe_allow_html=True)

# --------------------------
# TAB 6: HEALTH (FIXED FOR BANKS)
# --------------------------
elif nav == "🏥 Financial Health (Z-Score)":
    st.title("🏥 Financial Health")
    if 'ticker_data' not in st.session_state:
        st.warning("⚠️ Fetch ticker in Tab 2 first.")
    else:
        d = st.session_state['ticker_data']
        sector = d.get('sector', 'Unknown')
        name = d.get('name', '')
        
        # Check if Bank/Financial
        is_bank = 'Bank' in sector or 'Financial' in sector or 'Bank' in name
        
        if is_bank:
            st.error("⚠️ **Sector Restriction:** Altman Z-Score is NOT applicable for Banks & Financial Institutions.")
            st.info("""
            **Why?**
            - Banks carry high liabilities (deposits) which penalize the Z-Score unfairly.
            - A low score (like 0.11) for a bank does NOT mean bankruptcy.
            
            **Suggested Metrics for Banks:**
            - **Capital Adequacy Ratio (CAR)**
            - **Non-Performing Assets (NPA)**
            - **Net Interest Margin (NIM)**
            """)
        else:
            # Standard Calculation for Non-Banks
            try:
                ta, tl = d['total_assets'], d['total_liab']
                re, wc = d['retained_earnings'], d['working_capital']
                ebit, rev, mkt_cap = d['ebit'], d['revenue'], d['mkt_cap']
                
                if ta > 0:
                    z = (1.2*(wc/ta)) + (1.4*(re/ta)) + (3.3*(ebit/ta)) + (0.6*(mkt_cap/tl)) + (1.0*(rev/ta))
                    st.metric("Altman Z-Score", f"{z:.2f}")
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=z, title={'text': "Bankruptcy Risk"},
                        gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "black"},
                               'steps': [{'range': [0, 1.8], 'color': "#E74C3C"}, 
                                         {'range': [1.8, 3.0], 'color': "#F1C40F"}, 
                                         {'range': [3.0, 5.0], 'color': "#27AE60"}]}))
                    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if z > 3.0: st.success("✅ Safe Zone")
                    elif z > 1.8: st.warning("⚠️ Grey Zone (Caution)")
                    else: st.error("🚨 Distress Zone (High Risk)")
                else:
                    st.error("Insufficient Data for Z-Score.")
            except:
                st.error("Could not calculate Z-Score due to missing data.")

# --------------------------
# TAB 7: DEEP DIVE (FIXED FOR BANKS)
# --------------------------
elif nav == "📊 Deep Dive":
    st.title("📊 Fundamental Deep Dive")
    ticker_to_use = st.session_state.get('ticker_symbol', st.session_state.get('ticker_name'))
    
    if not ticker_to_use:
        st.warning("⚠️ Fetch ticker in Tab 2 first.")
    else:
        st.subheader(f"Historical Profitability: {ticker_to_use}")
        with st.spinner("Fetching data..."):
            df_margins = get_historical_profitability(ticker_to_use)
            if df_margins is not None and not df_margins.empty:
                fig_marg = px.line(df_margins, x=df_margins.index, y=df_margins.columns, title="Margin Trends", markers=True)
                fig_marg.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#E6D5B8"))
                st.plotly_chart(fig_marg, use_container_width=True)
                st.dataframe(df_margins.style.format("{:.2f}%"))
            else:
                st.warning("Could not calculate margins. (Note: Banks often lack 'Gross Margin' data in public feeds)")
