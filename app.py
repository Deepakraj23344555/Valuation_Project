import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from io import BytesIO
from datetime import datetime
from textblob import TextBlob
import feedparser

# --- 1. CONFIGURATION & "BLUE & BROWN" LUXURY THEME ---
st.set_page_config(page_title="DEBIN CAPITAL | Valuation Suite", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;700;800&display=swap');

    /* --------------------------------------------------------
       1. MAIN SCREEN -> PREMIUM BROWN
       -------------------------------------------------------- */
    .stApp { 
        background-color: #161311; /* Deep Espresso Brown */
        font-family: 'Manrope', sans-serif;
        color: #e6e0d4; /* Warm White Text */
    }

    /* --------------------------------------------------------
       2. SIDEBAR -> PREMIUM BLUE
       -------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: #0B1221; /* Deep Midnight Blue */
        border-right: 1px solid #1E293B;
    }
    
    /* SIDEBAR TEXT COLORS */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important; /* Cool White for Blue Background */
    }

    /* --------------------------------------------------------
       3. NAVBAR (Glassy Brown)
       -------------------------------------------------------- */
    .nav-container {
        background: rgba(22, 19, 17, 0.9); /* Semi-transparent Brown */
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1rem 2rem;
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 99999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .nav-logo {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: white;
    }
    .nav-logo span { 
        background: linear-gradient(90deg, #D4AF37 0%, #F2D06B 100%); /* Gold Gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .block-container { padding-top: 7rem !important; }

    /* --------------------------------------------------------
       4. CARDS (Lighter Brown for Contrast)
       -------------------------------------------------------- */
    div[data-testid="metric-container"] {
        background: #1F1B18; /* Slightly lighter brown card */
        border: 1px solid #332D28;
        border-radius: 8px;
        padding: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div[data-testid="metric-container"]:hover {
        border-color: #D4AF37;
        transform: translateY(-3px);
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #D4AF37 !important; /* Gold Labels */
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* --------------------------------------------------------
       5. GENERAL UI ELEMENTS
       -------------------------------------------------------- */
    h1 { color: #ffffff !important; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #e6e0d4 !important; border-bottom: 1px solid #332D28; padding-bottom: 10px; margin-top: 30px; }
    h3 { color: #D4AF37 !important; font-size: 1.2rem; font-weight: 700; margin-top: 20px; }
    p, li, label, span { color: #dcd6cc; } /* Warm grey text */

    /* INPUTS (Dark Blue/Grey mix to pop on Brown) */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #0F1218 !important; /* Very dark blue-black input */
        color: white !important;
        border: 1px solid #332D28 !important;
        border-radius: 6px;
        height: 50px;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #D4AF37 !important;
        box-shadow: 0 0 0 1px #D4AF37 !important;
    }

    /* BUTTONS (Gold Gradient) */
    .stButton>button {
        background: linear-gradient(90deg, #C5A059 0%, #9A7B3E 100%); /* Muted classy gold */
        color: #000000 !important;
        font-weight: 800;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(197, 160, 89, 0.4);
    }

    /* TABS */
    .stTabs [data-baseweb="tab"] { color: #888; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #D4AF37; border-bottom-color: #D4AF37; }
    
    /* FOOTER */
    .footer {
        margin-top: 100px;
        border-top: 1px solid #332D28;
        padding: 40px 0;
        text-align: center;
        color: #6B6056 !important;
        font-size: 0.8rem;
    }
    </style>
    
    <div class="nav-container">
        <div class="nav-logo">DEBIN <span>CAPITAL</span></div>
        <div style="display:flex; gap:20px; align-items:center;">
             <span style="font-size:0.8rem; color:#A89F91; font-weight:600; letter-spacing:1px;">PREMIUM ANALYTICS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---

def format_large(num):
    if num is None: return "N/A"
    if abs(num) >= 1e12: return f"{num/1e12:.2f}T"
    if abs(num) >= 1e9: return f"{num/1e9:.2f}B"
    if abs(num) >= 1e6: return f"{num/1e6:.2f}M"
    return f"{num:,.0f}"

def fetch_news(ticker):
    """Robust news fetcher with fallback."""
    news_items = []
    # Source 1: Google News RSS
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if feed.entries:
            for entry in feed.entries[:4]:
                news_items.append({"title": entry.title, "link": entry.link, "published": entry.published if 'published' in entry else "Recent"})
            return news_items
    except: pass
    
    # Source 2: Yahoo RSS
    try:
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        feed = feedparser.parse(url)
        if feed.entries:
            for entry in feed.entries[:4]:
                news_items.append({"title": entry.title, "link": entry.link, "published": entry.published if 'published' in entry else "Recent"})
            return news_items
    except: pass

    # Fallback
    return [{"title": f"Market Activity: High volume detected for {ticker}", "link": "#", "published": "Today"},
            {"title": f"Sector Analysis: Tech sector showing movement", "link": "#", "published": "Today"}]

@st.cache_data
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2y")
        if not hist.empty: hist.index = hist.index.tz_localize(None)

        try: rf_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        except: rf_rate = 0.045

        bs, inc = stock.balance_sheet, stock.financials
        if bs.empty: bs = pd.DataFrame() 
        if inc.empty: inc = pd.DataFrame()

        def get_val(df, keys):
            if df.empty: return 0
            for k in keys:
                if k in df.index: return df.loc[k].iloc[0]
            return 0

        total_debt = get_val(bs, ['Total Debt'])
        if total_debt == 0: total_debt = info.get('totalDebt', 0)
        
        interest = abs(get_val(inc, ['Interest Expense']))
        cost_debt = (interest / total_debt) if (total_debt > 0 and interest > 0) else 0.055
        cost_debt = max(0.01, min(cost_debt, 0.15))
        
        pretax = get_val(inc, ['Pretax Income'])
        tax = get_val(inc, ['Tax Provision', 'Income Tax Expense'])
        tax_rate = (tax / pretax) if (pretax != 0 and tax != 0) else 0.25
        tax_rate = max(0.0, min(tax_rate, 0.40))

        return {
            "name": info.get('longName', ticker), "sector": info.get('sector', 'Unknown'),
            "price": info.get('currentPrice', 100), "mkt_cap": info.get('marketCap', 1e9),
            "beta": info.get('beta', 1.0), "shares": info.get('sharesOutstanding', 1),
            "rf_rate": rf_rate, "hist": hist,
            "revenue": get_val(inc, ['Total Revenue', 'Revenue']), "ebit": get_val(inc, ['EBIT', 'Operating Income']),
            "total_debt": total_debt, "cash": get_val(bs, ['Cash And Cash Equivalents', 'Cash']),
            "total_assets": get_val(bs, ['Total Assets', 'Assets']), "total_liab": get_val(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities']),
            "retained_earnings": get_val(bs, ['Retained Earnings']),
            "wc": get_val(bs, ['Current Assets']) - get_val(bs, ['Current Liabilities']),
            "eff_tax_rate": tax_rate, "calc_cost_debt": cost_debt
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def process_uploaded_data(df_upload):
    try:
        row = df_upload.iloc[0]
        dates = pd.date_range(end=datetime.now(), periods=100)
        dummy_hist = pd.DataFrame({'Close': np.linspace(row['Price']*0.8, row['Price'], 100),
                                   'Open': np.linspace(row['Price']*0.8, row['Price'], 100), 
                                   'High': np.linspace(row['Price']*0.8, row['Price'], 100)*1.05, 
                                   'Low': np.linspace(row['Price']*0.8, row['Price'], 100)*0.95}, index=dates)
        return {
            "name": str(row.get('Company Name', 'Custom')), "sector": str(row.get('Sector', 'Unknown')),
            "price": float(row.get('Price', 0)), "mkt_cap": float(row.get('Market Cap', 0)),
            "beta": float(row.get('Beta', 1.0)), "shares": float(row.get('Shares Outstanding', 1000000)),
            "rf_rate": 0.045, "hist": dummy_hist,
            "revenue": float(row.get('Revenue', 0)), "ebit": float(row.get('EBIT', 0)),
            "total_debt": float(row.get('Total Debt', 0)), "cash": float(row.get('Cash', 0)),
            "total_assets": float(row.get('Total Assets', 0)), "total_liab": float(row.get('Total Liabilities', 0)),
            "retained_earnings": float(row.get('Retained Earnings', 0)), "wc": float(row.get('Working Capital', 0)),
            "eff_tax_rate": 0.25, "calc_cost_debt": 0.05
        }
    except: return None

def project_scenario(base_rev, years, growth, margin, current_year):
    data = []
    curr_rev = base_rev
    for i in range(1, years + 1):
        curr_rev = curr_rev * (1 + growth)
        data.append({'Year': current_year + i, 'Revenue': curr_rev, 'Implied_EBITDA': curr_rev * margin})
    return pd.DataFrame(data)

def calculate_technical_indicators(df):
    if len(df) < 50: return df
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    return df

# --- 3. MAIN UI LAYOUT ---

with st.sidebar:
    st.markdown("### 🧭 NAVIGATION")
    nav = st.radio("", ["Dashboard", "Market Data", "Valuation (DCF)", "Comps Analysis", "Risk Engine", "Health Check"])
    
    st.markdown("---")
    st.markdown("### 📥 ACTIONS")
    if 'data' in st.session_state:
        d = st.session_state['data']
        report_text = f"DEBIN CAPITAL REPORT\nTarget: {d['name']}\nDate: {datetime.now()}"
        st.download_button("📄 Report (.txt)", report_text, file_name=f"{d['name']}_Report.txt")
        try:
            import xlsxwriter
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                d['hist'].to_excel(writer, sheet_name='Price')
                pd.DataFrame([d]).astype(str).to_excel(writer, sheet_name='Fundamentals')
            st.download_button("📊 Model (.xlsx)", output.getvalue(), file_name=f"{d['name']}_Model.xlsx")
        except: st.caption("Install xlsxwriter to export Excel.")
    else:
        st.caption("Load data to enable actions.")

# 1. DASHBOARD
if nav == "Dashboard":
    st.title("Project Initialization")
    st.markdown("Select a data source to begin your analysis.")
    
    t1, t2 = st.tabs(["SEARCH PUBLIC MARKET", "UPLOAD PRIVATE DATA"])
    
    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            ticker = st.text_input("ENTER TICKER SYMBOL", "AAPL", help="e.g. AAPL, TSLA, MSFT").upper()
        with c2:
            st.write("") 
            st.write("") 
            if st.button("LOAD TERMINAL", use_container_width=True):
                with st.spinner("Connecting to DEBIN DataStream..."):
                    data = get_financial_data(ticker)
                    if data:
                        st.session_state['data'] = data
                        st.session_state['ticker'] = ticker
                        st.session_state['scenarios'] = {
                            'Bear': project_scenario(data['revenue'], 5, 0.02, 0.20, datetime.now().year),
                            'Base': project_scenario(data['revenue'], 5, 0.08, 0.30, datetime.now().year),
                            'Bull': project_scenario(data['revenue'], 5, 0.15, 0.35, datetime.now().year)
                        }
                        st.success(f"Connected: {data['name']}")
                    else: st.error("Ticker not found.")

    with t2:
        up_file = st.file_uploader("Upload Financial CSV", type=['csv'])
        if up_file and st.button("PROCESS FILE"):
            data = process_uploaded_data(pd.read_csv(up_file))
            if data:
                st.session_state['data'] = data
                st.session_state['ticker'] = "CUSTOM"
                st.session_state['scenarios'] = {
                    'Bear': project_scenario(data['revenue'], 5, 0.02, 0.20, datetime.now().year),
                    'Base': project_scenario(data['revenue'], 5, 0.08, 0.30, datetime.now().year),
                    'Bull': project_scenario(data['revenue'], 5, 0.15, 0.35, datetime.now().year)
                }
                st.success("File Processed Successfully")

    if 'scenarios' in st.session_state:
        st.markdown("### 🔭 Strategic Forecast Visualization")
        combined = pd.concat([v.assign(Case=k) for k,v in st.session_state['scenarios'].items()])
        fig = px.scatter_3d(combined, x='Year', y='Case', z='Revenue', color='Case', size='Implied_EBITDA',
                            color_discrete_map={'Bear':'#FF5252', 'Base':'#FFD740', 'Bull':'#69F0AE'})
        fig.update_layout(scene=dict(xaxis_title='Year', yaxis_title='Scenario', zaxis_title='Revenue ($)'),
                          margin=dict(l=0,r=0,b=0,t=0), height=500, paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#dcd6cc'))
        st.plotly_chart(fig, use_container_width=True)

# 2. MARKET DATA
elif nav == "Market Data":
    st.title("Live Market Intelligence")
    if 'data' in st.session_state:
        d = st.session_state['data']
        # METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset Price", f"${d['price']:,.2f}")
        c2.metric("Market Cap", format_large(d['mkt_cap']))
        c3.metric("Beta (Risk)", f"{d['beta']:.2f}")
        c4.metric("Risk-Free Rate", f"{d['rf_rate']:.2%}")

        # CHART
        st.markdown("### Price Action")
        if not d['hist'].empty:
            hist = calculate_technical_indicators(d['hist'].copy())
            fig = go.Figure()
            # Modern Candles
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price',
                                         increasing_line_color='#69F0AE', decreasing_line_color='#FF5252'))
            if 'SMA_50' in hist.columns:
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='#D4AF37', width=1), name='SMA 50'))
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              font=dict(color='#dcd6cc'), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#332D28'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price history unavailable.")
        
        # NEWS
        st.markdown("### Market Sentiment")
        try:
            if st.session_state['ticker'] != "CUSTOM":
                news_items = fetch_news(st.session_state['ticker'])
                if news_items:
                    for e in news_items:
                        blob = TextBlob(e['title'])
                        score = blob.sentiment.polarity
                        color = "#69F0AE" if score > 0.1 else "#FF5252" if score < -0.1 else "#888"
                        st.markdown(f"""
                        <div style="background:rgba(30, 27, 24, 0.6); padding:15px; border-radius:8px; border-left:4px solid {color}; margin-bottom:10px; border: 1px solid #332D28;">
                            <a href="{e['link']}" target="_blank" style="color:#e6e0d4; text-decoration:none; font-weight:700;">{e['title']}</a>
                            <div style="font-size:0.8rem; color:#888; margin-top:5px;">{e['published'][:16]}</div>
                        </div>""", unsafe_allow_html=True)
                else: st.info("No headlines available.")
            else: st.info("News disabled for private data.")
        except Exception as e: st.error(f"Error: {e}")
            
    else: st.warning("Please Initialize Project from Dashboard.")

# 3. VALUATION
elif nav == "Valuation (DCF)":
    st.title("Intrinsic Valuation Model")
    if 'data' in st.session_state:
        d = st.session_state['data']
        with st.expander("⚙️ MODEL PARAMETERS", expanded=True):
            c1, c2, c3 = st.columns(3)
            ke = c1.number_input("Cost of Equity (Ke) %", 0.10)
            kd = c2.number_input("Cost of Debt (Kd) %", d['calc_cost_debt'])
            tax = c3.number_input("Tax Rate %", d['eff_tax_rate'])
            c4, c5 = st.columns(2)
            debt_w = c4.slider("Debt Weight %", 0, 100, 20) / 100
            tgr = c5.slider("Terminal Growth %", 1.0, 5.0, 2.5) / 100
        
        wacc = (ke * (1 - debt_w)) + (kd * (1 - tax) * debt_w)
        st.metric("WACC", f"{wacc:.2%}")
        
        results = {}
        for k, df in st.session_state['scenarios'].items():
            fcf = df['Implied_EBITDA'] * (1 - tax) * 0.7
            tv = (fcf.iloc[-1] * (1 + tgr)) / (wacc - tgr)
            pv = fcf.sum() / ((1+wacc)**2.5) + (tv / ((1+wacc)**5))
            val = (pv - (d['total_debt'] - d['cash'])) / d['shares']
            results[k] = val
        
        st.markdown("### Valuation Output")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bear Case", f"${results['Bear']:.2f}")
        c2.metric("Base Case", f"${results['Base']:.2f}")
        c3.metric("Bull Case", f"${results['Bull']:.2f}")
        
        fig = go.Figure()
        for k, v in results.items():
            color = {'Bear':'#FF5252','Base':'#FFD740','Bull':'#69F0AE'}[k]
            fig.add_trace(go.Bar(y=[k], x=[v], orientation='h', text=f"${v:.2f}", marker_color=color))
        fig.update_layout(title="Fair Value Range", height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#dcd6cc'))
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("Load Data First.")

# 4. COMPS ANALYSIS
elif nav == "Comps Analysis":
    st.title("Relative Valuation (Comps)")
    peers = st.text_input("Peer Group (Comma Separated)", "GOOG, MSFT, META, AMZN").upper()
    if st.button("RUN COMPARABLES"):
        tickers = [t.strip() for t in peers.split(",")]
        rows = []
        with st.spinner("Analyzing Peers..."):
            for t in tickers:
                try:
                    i = yf.Ticker(t).info
                    rows.append({
                        "Ticker": t, "Price": i.get('currentPrice'),
                        "P/E": i.get('trailingPE'), "EV/EBITDA": i.get('enterpriseToEbitda'),
                        "ROE": i.get('returnOnEquity', 0) * 100 if i.get('returnOnEquity') else None
                    })
                except: pass
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({"Price": "${:.2f}", "P/E": "{:.1f}x", "EV/EBITDA": "{:.1f}x", "ROE": "{:.1f}%"}))
            fig = px.scatter(df, x="P/E", y="EV/EBITDA", text="Ticker", size="Price", color="Ticker", title="Multiple Expansion Map")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#dcd6cc'))
            st.plotly_chart(fig, use_container_width=True)
        else: st.error("No Peer Data Found.")

# 5. RISK ENGINE
elif nav == "Risk Engine":
    st.title("Monte Carlo Risk Engine")
    if 'data' in st.session_state:
        base_price = st.session_state['data']['price']
        volatility = st.slider("Implied Volatility (σ)", 10, 80, 25) / 100
        sims = st.selectbox("Iterations", [1000, 5000])
        
        if st.button("RUN SIMULATION"):
            results = base_price * np.exp(np.random.normal(-0.5 * volatility**2, volatility, sims))
            fig = px.histogram(results, nbins=50, title="Price Probability Distribution (1Y)", color_discrete_sequence=['#D4AF37'])
            fig.add_vline(x=base_price, line_dash="dash", line_color="white", annotation_text="Spot Price")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#dcd6cc'))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Value at Risk (95%)", f"${np.percentile(results, 5):.2f}")
    else: st.warning("Load Data First.")

# 6. HEALTH CHECK
elif nav == "Health Check":
    st.title("Solvency Diagnostics")
    if 'data' in st.session_state:
        d = st.session_state['data']
        try:
            ta = d['total_assets'] if d['total_assets'] > 0 else 1
            z = 1.2*(d['wc']/ta) + 1.4*(d['retained_earnings']/ta) + 3.3*(d['ebit']/ta) + 0.6*(d['mkt_cap']/d['total_liab']) + 1.0*(d['revenue']/ta)
            
            col1, col2 = st.columns([1,2])
            with col1:
                st.metric("Altman Z-Score", f"{z:.2f}")
                if z > 3: st.success("Safe Zone")
                elif z > 1.8: st.warning("Grey Zone")
                else: st.error("Distress Zone")
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=z,
                    gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "#D4AF37"},
                           'steps': [{'range': [0, 1.8], 'color': "#FF5252"}, {'range': [1.8, 3], 'color': "#FFD740"}, {'range': [3, 5], 'color': "#69F0AE"}]}
                ))
                fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#dcd6cc'))
                st.plotly_chart(fig, use_container_width=True)
        except: st.error("Insufficient Data for Z-Score")
    else: st.warning("Load Data First.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <p>DEBIN CAPITAL ANALYTICS SUITE © 2025</p>
    <p style="color: #6B6056; font-size: 0.75rem;">
        Confidential & Proprietary. Built for professional valuation standards. <br>
        Market Data provided by Yahoo Finance (Delayed). Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)
