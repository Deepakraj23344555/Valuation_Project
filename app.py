import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from io import BytesIO
from datetime import datetime
from textblob import TextBlob
import time

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="GT Valuation Terminal", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    /* Main Theme */
    .stApp { background-color: #1e1e1e; }
    
    /* Typography */
    h1, h2, h3 { color: #d4af37 !important; font-family: 'Segoe UI', sans-serif; }
    p, li, span, label { color: #e0e0e0; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid #333; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #d4af37; font-size: 28px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #a0a0a0; font-size: 14px; }
    
    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input { 
        color: black !important; background-color: #f0f2f6 !important; font-weight: bold; 
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #d4af37; color: #121212 !important;
        border: none; border-radius: 4px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #f1c40f; }
    
    /* Tables */
    [data-testid="stDataFrame"] { border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (ROBUST & MOCK FALLBACK) ---

def format_large(num):
    if num is None or num == 0: return "N/A"
    if abs(num) >= 1e12: return f"{num/1e12:.2f}T"
    if abs(num) >= 1e9: return f"{num/1e9:.2f}B"
    if abs(num) >= 1e6: return f"{num/1e6:.2f}M"
    return f"{num:,.0f}"

def get_mock_data():
    """
    Returns specific static data for AAPL to ensure the app works 
    even when Yahoo Finance rate-limits the server.
    """
    # Create mock margin dataframe
    dates = pd.date_range(end=datetime.today(), periods=4, freq='YE')
    margins = pd.DataFrame({
        'Gross_Margin': [43.0, 43.5, 44.0, 45.0],
        'Operating_Margin': [28.0, 29.0, 30.0, 30.5],
        'Net_Margin': [24.0, 25.0, 25.5, 26.0]
    }, index=dates)
    
    # Create mock history
    hist_dates = pd.date_range(end=datetime.today(), periods=500)
    hist = pd.DataFrame({
        'Open': np.linspace(150, 230, 500),
        'High': np.linspace(155, 235, 500),
        'Low': np.linspace(145, 225, 500),
        'Close': np.linspace(150, 230, 500) + np.random.normal(0, 5, 500)
    }, index=hist_dates)

    return {
        "valid": True,
        "is_mock": True,
        "name": "Apple Inc. (DEMO MODE)",
        "sector": "Technology",
        "summary": "This is cached demo data displayed because Yahoo Finance is temporarily rate-limiting requests. Use this to test the app features.",
        "currency": "USD",
        "price": 230.00,
        "mkt_cap": 3500000000000,
        "beta": 1.12,
        "shares": 15000000000,
        "rf_rate": 0.045,
        "hist": hist,
        "revenue": 383000000000,
        "ebit": 114000000000,
        "total_debt": 100000000000,
        "cash": 30000000000,
        "total_assets": 352000000000,
        "total_liab": 290000000000,
        "retained_earnings": 0, # Apple often has 0 here due to buybacks
        "wc": -5000000000,
        "eff_tax_rate": 0.15,
        "calc_cost_debt": 0.045,
        "margins_df": margins
    }

@st.cache_data(ttl=3600)
def get_all_data(ticker):
    """
    Master function to fetch ALL data at once. 
    Includes fallback to Mock Data if Yahoo Finance fails.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Check connection by fetching info
        info = stock.info
        if not info or 'regularMarketPrice' not in info and 'currentPrice' not in info:
            raise ValueError("No data found")

        # 2. History (Prices)
        hist = stock.history(period="2y")
        
        # 3. Financial Statements
        bs = stock.balance_sheet
        inc = stock.financials
        
        # 4. Market Data
        try:
            rf_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        except:
            rf_rate = 0.045

        # --- Helper ---
        def get_item(df, keys):
            if df.empty: return 0
            for k in keys:
                if k in df.index: return df.loc[k].iloc[0]
            return 0

        # --- EXTRACT METRICS ---
        total_assets = get_item(bs, ['Total Assets', 'Assets'])
        total_liab = get_item(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
        retained_earnings = get_item(bs, ['Retained Earnings'])
        curr_assets = get_item(bs, ['Current Assets'])
        curr_liab = get_item(bs, ['Current Liabilities'])
        
        revenue = get_item(inc, ['Total Revenue', 'Revenue', 'Total Income']) 
        ebit = get_item(inc, ['EBIT', 'Operating Income', 'Pretax Income']) 
        interest_exp = abs(get_item(inc, ['Interest Expense', 'Interest Expense Non Operating']))
        tax_exp = get_item(inc, ['Tax Provision', 'Income Tax Expense'])
        pretax_inc = get_item(inc, ['Pretax Income'])
        
        total_debt = get_item(bs, ['Total Debt'])
        if total_debt == 0: total_debt = info.get('totalDebt', 0)
        cash = get_item(bs, ['Cash And Cash Equivalents', 'Cash', 'Cash & Equivalents'])

        # Margins (Fault Tolerant)
        margins_df = None
        if not inc.empty:
            margins_df = pd.DataFrame(index=inc.columns)
            rev_series = None
            for key in ['Total Revenue', 'Revenue', 'Total Income']:
                if key in inc.index:
                    rev_series = inc.loc[key]
                    break
            
            if rev_series is not None:
                if 'Gross Profit' in inc.index:
                    margins_df['Gross_Margin'] = (inc.loc['Gross Profit'] / rev_series) * 100
                else:
                    margins_df['Gross_Margin'] = np.nan 
                
                op_idx = next((k for k in ['Operating Income', 'EBIT', 'Pretax Income'] if k in inc.index), None)
                if op_idx:
                    margins_df['Operating_Margin'] = (inc.loc[op_idx] / rev_series) * 100
                else:
                    margins_df['Operating_Margin'] = np.nan
                
                if 'Net Income' in inc.index:
                    margins_df['Net_Margin'] = (inc.loc['Net Income'] / rev_series) * 100
                else:
                    margins_df['Net_Margin'] = np.nan
                
                margins_df = margins_df.sort_index()

        # Defaults
        eff_tax_rate = 0.25
        if pretax_inc != 0:
            eff_tax_rate = max(0.0, min(tax_exp / pretax_inc, 0.40))
            
        cost_debt = 0.055
        if total_debt > 0 and interest_exp > 0:
            cost_debt = max(0.01, min(interest_exp / total_debt, 0.15))

        return {
            "valid": True,
            "is_mock": False,
            "name": info.get('longName', ticker),
            "sector": info.get('sector', 'Unknown'),
            "summary": info.get('longBusinessSummary', 'No description.'),
            "currency": info.get('currency', 'USD'),
            "price": info.get('currentPrice', 0),
            "mkt_cap": info.get('marketCap', 0),
            "beta": info.get('beta', 1.0),
            "shares": info.get('sharesOutstanding', 1),
            "rf_rate": rf_rate,
            "hist": hist,
            "revenue": revenue,
            "ebit": ebit,
            "total_debt": total_debt,
            "cash": cash,
            "total_assets": total_assets,
            "total_liab": total_liab,
            "retained_earnings": retained_earnings,
            "wc": curr_assets - curr_liab,
            "eff_tax_rate": eff_tax_rate,
            "calc_cost_debt": cost_debt,
            "margins_df": margins_df
        }

    except Exception as e:
        # FAIL-SAFE: Return Mock Data instead of crashing
        print(f"API Failed: {e}. Switching to Mock Data.")
        return get_mock_data()

def project_scenario(base_rev, years, growth, margin, current_year):
    data = []
    curr_rev = base_rev
    for i in range(1, years + 1):
        curr_rev = curr_rev * (1 + growth)
        data.append({
            'Year': current_year + i,
            'Revenue': curr_rev,
            'EBITDA_Margin': margin,
            'Implied_EBITDA': curr_rev * margin
        })
    return pd.DataFrame(data)

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
        display_df = base_df[['Year', 'Revenue', 'EBITDA_Margin', 'Implied_EBITDA']].copy()
        forecast_html = display_df.to_html(classes='table', index=False, float_format=lambda x: f"{x:,.0f}" if x > 1 else f"{x:.1%}")

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

# --- 3. APP LAYOUT ---

with st.sidebar:
    st.title("GT Terminal 🚀")
    st.caption("Bulletproof Edition v12.0")
    st.markdown("---")
    nav = st.radio("Modules", [
        "Project Setup", 
        "Live Market Data", 
        "Valuation Model (DCF)", 
        "Peer Analysis", 
        "Risk Simulation", 
        "Financial Health",
        "Deep Dive"
    ])
    st.markdown("---")

# ----------------------------
# 1. PROJECT SETUP
# ----------------------------
if nav == "Project Setup":
    st.title("📂 Project Initialization")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("Enter ticker. The app will fetch data once. If Rate Limited, it loads DEMO data.")
        ticker_input = st.text_input("Ticker Symbol", "AAPL").upper()
        
    if st.button("🚀 Initialize Model"):
        with st.spinner(f"Analyzing {ticker_input}..."):
            # Master Fetch
            data = get_all_data(ticker_input)
            
            st.session_state['data'] = data
            st.session_state['ticker'] = ticker_input
            
            if data.get('is_mock'):
                st.warning("⚠️ **RATE LIMIT DETECTED:** Yahoo Finance blocked the request. Loaded **DEMO DATA** (Apple Inc) so you can still use the app.")
            else:
                st.success(f"Successfully loaded live data for {data['name']}")
            
            # Auto-build Scenarios
            base_rev = data['revenue']
            if base_rev == 0: base_rev = 1e9 # Fallback
                
            year = datetime.now().year
            scenarios = {
                'Bear': project_scenario(base_rev, 5, 0.02, 0.20, year),
                'Base': project_scenario(base_rev, 5, 0.08, 0.30, year),
                'Bull': project_scenario(base_rev, 5, 0.15, 0.35, year)
            }
            st.session_state['scenarios'] = scenarios

    if 'scenarios' in st.session_state:
        st.markdown("### 📊 Scenario Preview")
        combined = pd.DataFrame()
        for k, v in st.session_state['scenarios'].items():
            temp = v.copy()
            temp['Case'] = k
            combined = pd.concat([combined, temp])
            
        fig = px.bar(combined, x='Year', y='Revenue', color='Case', barmode='group',
                     color_discrete_map={'Bear':'#ef553b', 'Base':'#f1c40f', 'Bull':'#00cc96'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 2. LIVE MARKET
# ----------------------------
elif nav == "Live Market Data":
    st.title("📈 Market Terminal")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize in 'Project Setup' first.")
    else:
        d = st.session_state['data']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"${d['price']:,.2f}")
        m2.metric("Market Cap", format_large(d['mkt_cap']))
        m3.metric("Beta", f"{d['beta']:.2f}")
        m4.metric("Risk-Free", f"{d['rf_rate']:.2%}")
        
        st.subheader("Price Action")
        if not d['hist'].empty:
            hist = d['hist'].copy()
            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                         low=hist['Low'], close=hist['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='#f1c40f'), name='SMA 50'))
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#e0e0e0'), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No price history available.")

# ----------------------------
# 3. VALUATION (DCF)
# ----------------------------
elif nav == "Valuation Model (DCF)":
    st.title("💎 Intrinsic Valuation")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize in 'Project Setup' first.")
    else:
        d = st.session_state['data']
        
        with st.expander("⚙️ Model Inputs", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1: cost_equity = st.number_input("Cost of Equity", 0.10, step=0.005)
            with c2: cost_debt = st.number_input("Cost of Debt", d['calc_cost_debt'], step=0.005)
            with c3: tax_rate = st.number_input("Tax Rate", d['eff_tax_rate'], step=0.01)
            
            c4, c5 = st.columns(2)
            with c4: debt_weight = st.slider("Debt Weight", 0.0, 1.0, 0.2)
            with c5: tgr = st.slider("Terminal Growth", 0.01, 0.05, 0.025)

        wacc = (cost_equity * (1 - debt_weight)) + (cost_debt * (1 - tax_rate) * debt_weight)
        st.metric("WACC", f"{wacc:.2%}")
        
        results = {}
        if 'scenarios' in st.session_state:
            for name, df in st.session_state['scenarios'].items():
                df['FCF_Proxy'] = df['Implied_EBITDA'] * (1 - tax_rate) * 0.7
                df['PV'] = df['FCF_Proxy'] / [(1+wacc)**i for i in range(1,6)]
                
                tv = (df['FCF_Proxy'].iloc[-1] * (1+tgr)) / (wacc-tgr)
                pv_tv = tv / ((1+wacc)**5)
                
                ev = df['PV'].sum() + pv_tv
                net_debt = d['total_debt'] - d['cash']
                equity = ev - net_debt
                shares_out = d['shares'] if d['shares'] > 0 else 1
                share_price = equity / shares_out
                results[name] = max(0, share_price) 
                
                if name == 'Base': st.session_state['base_df'] = df

            st.subheader("🎯 Implied Share Price")
            c1, c2, c3 = st.columns(3)
            c1.metric("Bear", f"${results.get('Bear', 0):.2f}")
            c2.metric("Base", f"${results.get('Base', 0):.2f}")
            c3.metric("Bull", f"${results.get('Bull', 0):.2f}")

            # 2D Matrix
            if 'base_df' in st.session_state:
                st.subheader("Sensitivity Analysis (Base Case)")
                wacc_range = np.linspace(wacc-0.02, wacc+0.02, 5)
                tgr_range = np.linspace(tgr-0.01, tgr+0.01, 5)
                
                matrix = []
                for t in tgr_range:
                    row = []
                    for w in wacc_range:
                        if w == t: w += 0.001 
                        last_fcf = st.session_state['base_df']['FCF_Proxy'].iloc[-1]
                        pv_explicit = st.session_state['base_df']['PV'].sum()
                        tv = (last_fcf * (1 + t)) / (w - t)
                        val = (pv_explicit + (tv / ((1+w)**5)) - net_debt) / d['shares']
                        row.append(max(0, val))
                    matrix.append(row)
                    
                df_mat = pd.DataFrame(matrix, index=[f"G: {x:.1%}" for x in tgr_range], 
                                      columns=[f"W: {x:.1%}" for x in wacc_range])
                st.dataframe(df_mat.style.format("${:.2f}").background_gradient(cmap='RdYlGn', axis=None))

# ----------------------------
# 4. FINANCIAL HEALTH
# ----------------------------
elif nav == "Financial Health":
    st.title("🏥 Financial Health Check")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize in 'Project Setup' first.")
    else:
        d = st.session_state['data']
        
        # Sector Logic
        is_bank = 'Financial' in d['sector'] or 'Bank' in d['name'] or 'Capital' in d['name']
        
        if is_bank:
            st.warning(f"⚠️ **Sector Notice:** {d['name']} is a Financial Institution.")
            st.info("Altman Z-Score is NOT valid for banks due to high liability structures (deposits).")
        else:
            try:
                # Z-Score Calculation (Only if not a bank)
                if d['total_assets'] > 0 and d['total_liab'] > 0:
                    A = d['wc'] / d['total_assets']
                    B = d['retained_earnings'] / d['total_assets']
                    C = d['ebit'] / d['total_assets']
                    D = d['mkt_cap'] / d['total_liab']
                    E = d['revenue'] / d['total_assets']
                    
                    z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
                    
                    st.metric("Altman Z-Score", f"{z:.2f}")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=z,
                        gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "white"},
                               'steps': [{'range': [0, 1.8], 'color': "#ef553b"},
                                         {'range': [1.8, 3], 'color': "#f1c40f"},
                                         {'range': [3, 5], 'color': "#00cc96"}]}
                    ))
                    fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if z > 3: st.success("Safe Zone")
                    elif z > 1.8: st.warning("Grey Zone")
                    else: st.error("Distress Zone")
                else:
                    st.error("Insufficient Balance Sheet data for Z-Score.")
            except:
                st.error("Error calculating Z-Score.")

# ----------------------------
# 5. DEEP DIVE
# ----------------------------
elif nav == "Deep Dive":
    st.title("📊 Fundamental Deep Dive")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize in 'Project Setup' first.")
    else:
        d = st.session_state['data']
        st.subheader(f"Historical Profitability: {d['name']}")
        
        df_margins = d.get('margins_df')
        
        if df_margins is not None and not df_margins.empty:
            # Drop columns that are all NaN (e.g. Gross Margin for Banks)
            df_plot = df_margins.dropna(axis=1, how='all')
            
            if not df_plot.empty:
                fig = px.line(df_plot, x=df_plot.index, y=df_plot.columns, markers=True,
                              title="Margin Trends (%)")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_plot.style.format("{:.2f}%"))
            else:
                st.warning("Margins could not be calculated (Data might be missing).")
        else:
            st.error("Historical financial data unavailable.")

# ----------------------------
# 6. RISK SIMULATION
# ----------------------------
elif nav == "Risk Simulation":
    st.title("🎲 Monte Carlo")
    if 'data' not in st.session_state:
        st.warning("Please initialize in 'Project Setup' first.")
    else:
        curr_price = st.session_state['data']['price']
        vol = st.slider("Volatility", 0.1, 0.5, 0.2)
        sims = st.selectbox("Iterations", [1000, 5000])
        
        if st.button("Run"):
            results = curr_price * np.exp(np.random.normal(-0.5*vol**2, vol, sims))
            fig = px.histogram(results, title="Price Distribution (1 Yr)", color_discrete_sequence=['#d4af37'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 7. PEER ANALYSIS
# ----------------------------
elif nav == "Peer Analysis":
    st.title("🌍 Peer Comparison")
    peers = st.text_input("Enter Peers (comma separated)", "GOOG, MSFT, AMZN")
    
    if st.button("Compare"):
        tickers = [t.strip().upper() for t in peers.split(",")]
        rows = []
        with st.spinner("Fetching peers..."):
            for t in tickers:
                try:
                    i = yf.Ticker(t).info
                    rows.append({
                        "Ticker": t,
                        "P/E": i.get('trailingPE', 0),
                        "ROE": i.get('returnOnEquity', 0),
                        "Debt/Eq": i.get('debtToEquity', 0)
                    })
                except: pass
        
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
        else:
            st.error("No peer data found.")
