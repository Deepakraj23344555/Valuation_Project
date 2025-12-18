import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from io import BytesIO
from datetime import datetime
from textblob import TextBlob

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="GT Valuation Terminal", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(to bottom right, #0f0c29, #302b63, #24243e); color: white; }
    section[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #d4af37; }
    h1, h2, h3 { color: #d4af37 !important; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        background: linear-gradient(90deg, #d4af37, #f1c40f); color: black !important;
        font-weight: bold; border: none; border-radius: 20px;
    }
    .metric-card {
        background: rgba(255,255,255,0.05); border: 1px solid #d4af37;
        border-radius: 10px; padding: 15px; text-align: center;
    }
    .stTextInput>div>div>input { color: black !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (WITH MANUAL & AUTO FALLBACK) ---

def get_mock_data():
    """Returns static data to bypass API limits."""
    # Mock History
    dates = pd.date_range(end=datetime.today(), periods=300)
    hist = pd.DataFrame({
        'Open': np.linspace(150, 230, 300),
        'Close': np.linspace(150, 230, 300) + np.random.normal(0, 2, 300)
    }, index=dates)
    
    # Mock Margins
    margin_dates = pd.date_range(end=datetime.today(), periods=4, freq='YE')
    margins = pd.DataFrame({
        'Gross_Margin': [40, 42, 41, 43],
        'Operating_Margin': [25, 26, 25, 28],
        'Net_Margin': [20, 21, 20, 22]
    }, index=margin_dates)

    return {
        "valid": True,
        "is_mock": True,
        "ticker": "AAPL",
        "name": "Apple Inc. (DEMO MODE)",
        "sector": "Technology",
        "summary": "This is cached demo data loaded because Yahoo Finance is rate-limiting your requests.",
        "price": 230.50,
        "mkt_cap": 3500000000000,
        "beta": 1.15,
        "pe": 32.5,
        "rf_rate": 0.045,
        "hist": hist,
        "revenue": 383000000000,
        "net_income": 97000000000,
        "op_cash_flow": 110000000000,
        "total_assets": 352000000000,
        "total_liab": 290000000000,
        "long_term_debt": 95000000000,
        "cash": 30000000000,
        "shares": 15000000000,
        "current_ratio": 1.05,
        "margins_df": margins,
        # F-Score specific
        "prev_net_income": 90000000000,
        "prev_assets": 330000000000,
        "prev_long_term_debt": 98000000000,
        "prev_shares": 15500000000,
        "gross_profit": 170000000000,
        "prev_gross_profit": 160000000000
    }

@st.cache_data(ttl=3600)
def get_live_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Validation: Check if data exists
        if not info or 'regularMarketPrice' not in info and 'currentPrice' not in info:
            raise ValueError("No live data found")

        hist = stock.history(period="2y")
        bs = stock.balance_sheet
        inc = stock.financials
        cf = stock.cashflow
        
        # Helper to safely get values
        def get(df, keys):
            if df.empty: return 0
            for k in keys:
                if k in df.index: return df.loc[k].iloc[0]
            return 0
            
        def get_prev(df, keys):
            if df.empty or df.shape[1] < 2: return 0
            for k in keys:
                if k in df.index: return df.loc[k].iloc[1]
            return 0

        # Extract Metrics
        revenue = get(inc, ['Total Revenue', 'Revenue', 'Total Income'])
        
        # Margins Logic
        margins_df = pd.DataFrame()
        if not inc.empty and revenue > 0:
            margins_df = pd.DataFrame(index=inc.columns)
            if 'Gross Profit' in inc.index:
                margins_df['Gross_Margin'] = (inc.loc['Gross Profit'] / inc.loc['Total Revenue']) * 100
            if 'Net Income' in inc.index:
                margins_df['Net_Margin'] = (inc.loc['Net Income'] / inc.loc['Total Revenue']) * 100
            margins_df = margins_df.sort_index()

        return {
            "valid": True,
            "is_mock": False,
            "ticker": ticker,
            "name": info.get('longName', ticker),
            "sector": info.get('sector', 'Unknown'),
            "summary": info.get('longBusinessSummary', 'No summary.'),
            "price": info.get('currentPrice', 0),
            "mkt_cap": info.get('marketCap', 0),
            "beta": info.get('beta', 1.0),
            "pe": info.get('trailingPE', 0),
            "rf_rate": 0.045,
            "hist": hist,
            "revenue": revenue,
            "net_income": get(inc, ['Net Income', 'Net Income Common Stockholders']),
            "prev_net_income": get_prev(inc, ['Net Income']),
            "op_cash_flow": get(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities']),
            "total_assets": get(bs, ['Total Assets']),
            "prev_assets": get_prev(bs, ['Total Assets']),
            "total_liab": get(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities']),
            "long_term_debt": get(bs, ['Long Term Debt']),
            "prev_long_term_debt": get_prev(bs, ['Long Term Debt']),
            "cash": get(bs, ['Cash', 'Cash And Cash Equivalents']),
            "shares": info.get('sharesOutstanding', 1),
            "prev_shares": get_prev(bs, ['Share Issued']),
            "current_ratio": info.get('currentRatio', 0),
            "gross_profit": get(inc, ['Gross Profit']),
            "prev_gross_profit": get_prev(inc, ['Gross Profit']),
            "margins_df": margins_df
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

def calculate_piotroski(d):
    score = 0
    # Profitability
    if d['net_income'] > 0: score += 1
    if d['op_cash_flow'] > 0: score += 1
    try:
        roa = d['net_income'] / d['total_assets']
        prev_roa = d['prev_net_income'] / d['prev_assets']
        if roa > prev_roa: score += 1
    except: pass
    if d['op_cash_flow'] > d['net_income']: score += 1
    
    # Leverage
    if d['long_term_debt'] < d['prev_long_term_debt']: score += 1
    if d['current_ratio'] > 1: score += 1
    if d['shares'] <= d['prev_shares']: score += 1
    
    # Efficiency
    try:
        gm = d['gross_profit'] / d['revenue']
        prev_gm = d['prev_gross_profit'] / (d['revenue'] * 0.9) # proxy
        if gm > prev_gm: score += 1
    except: pass
    
    return score

# --- 3. UI LAYOUT ---

with st.sidebar:
    st.title("🏛️ Titan Terminal")
    st.caption("v13.0 | Offline-Ready")
    st.markdown("---")
    
    # FORCE DEMO MODE SWITCH
    use_demo = st.checkbox("⚠️ Force Demo Mode (Offline)", value=False, help="Check this if Yahoo Finance is giving errors.")
    
    nav = st.radio("Navigation", ["Dashboard", "Intrinsic Valuation", "F-Score (Health)", "Technicals"])
    st.markdown("---")

# ==========================================
# 1. DASHBOARD
# ==========================================
if nav == "Dashboard":
    st.title("🚀 Executive Dashboard")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker_input = st.text_input("Enter Ticker", "AAPL").upper()
        if st.button("Initialize Analysis"):
            with st.spinner("Processing..."):
                if use_demo:
                    data = get_mock_data()
                    st.success("Loaded Demo Data (Offline Mode)")
                else:
                    data = get_live_data(ticker_input)
                    # Auto-fallback if live fails
                    if not data['valid']:
                        st.error(f"API Error: {data.get('error')}. Switching to Demo Data.")
                        data = get_mock_data()
                    else:
                        st.success(f"Loaded Live Data for {ticker_input}")
                
                st.session_state['data'] = data

    if 'data' in st.session_state:
        d = st.session_state['data']
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><h3>Price</h3><h2>${d['price']:,.2f}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Market Cap</h3><h2>${d['mkt_cap']/1e9:,.1f}B</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Beta</h3><h2>{d['beta']:.2f}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><h3>P/E</h3><h2>{d['pe']:.1f}x</h2></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🤖 AI Analyst Summary")
        
        # Simple Logic for Summary
        valuation = "undervalued" if d['pe'] > 0 and d['pe'] < 15 else "premium priced"
        risk = "low volatility" if d['beta'] < 1.0 else "high volatility"
        profit = "profitable" if d['net_income'] > 0 else "unprofitable"
        
        st.info(f"""
        **Executive Summary for {d['name']}:**
        The company is currently **{profit}** and trades at a **{valuation}** multiple relative to historical norms.
        The stock exhibits **{risk}**, making it suitable for specific portfolio strategies.
        
        *Sector:* {d['sector']}
        """)

# ==========================================
# 2. INTRINSIC VALUATION
# ==========================================
elif nav == "Intrinsic Valuation":
    st.title("💎 Discounted Cash Flow (DCF)")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize in Dashboard first.")
    else:
        d = st.session_state['data']
        
        c1, c2 = st.columns(2)
        with c1: growth = st.slider("Growth Rate %", 0, 30, 10) / 100
        with c2: wacc = st.slider("WACC %", 5, 15, 9) / 100
        
        # Math
        fcf = d['op_cash_flow'] if d['op_cash_flow'] > 0 else 1e9
        projections = [fcf * ((1 + growth) ** i) for i in range(1, 6)]
        
        term_val = (projections[-1] * 1.025) / (wacc - 0.025)
        pv_flows = sum([f / ((1 + wacc) ** (i+1)) for i, f in enumerate(projections)])
        pv_tv = term_val / ((1 + wacc) ** 5)
        
        ev = pv_flows + pv_tv
        equity = ev + d['cash'] - d['long_term_debt']
        target = equity / d['shares']
        
        st.markdown(f"""
        <div style="text-align:center; padding:20px; border:1px solid #d4af37; border-radius:10px;">
            <h2>Fair Value Estimate</h2>
            <h1 style="color:#d4af37; font-size:50px;">${target:,.2f}</h1>
            <p>Current: ${d['price']:.2f} | Upside: {((target-d['price'])/d['price'])*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 3. F-SCORE (HEALTH)
# ==========================================
elif nav == "F-Score (Health)":
    st.title("💯 Piotroski F-Score")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize in Dashboard first.")
    else:
        d = st.session_state['data']
        score = calculate_piotroski(d)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = score,
            title = {'text': "Financial Strength (0-9)"},
            gauge = {
                'axis': {'range': [0, 9]},
                'bar': {'color': "#d4af37"},
                'steps': [
                    {'range': [0, 3], 'color': '#ef553b'},
                    {'range': [3, 6], 'color': '#f1c40f'},
                    {'range': [6, 9], 'color': '#00cc96'}]
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        if score >= 7: st.success("🌟 Elite Fundamentals")
        elif score >= 4: st.warning("⚖️ Average Fundamentals")
        else: st.error("🚩 Weak Fundamentals")

# ==========================================
# 4. TECHNICALS
# ==========================================
elif nav == "Technicals":
    st.title("📈 Technical Charts")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize in Dashboard first.")
    else:
        d = st.session_state['data']
        hist = d['hist'].copy()
        
        # SMA
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                     low=hist['Low'], close=hist['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='#f1c40f'), name='SMA 50'))
        
        fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font=dict(color='white'), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
