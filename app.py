import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from io import BytesIO
from datetime import datetime
from textblob import TextBlob

# --- 1. CONFIGURATION & STYLING (HIGH CONTRAST EDITION) ---
st.set_page_config(page_title="GT Valuation Terminal", page_icon="💎", layout="wide")

st.markdown("""
    <style>
    /* IMPORT GOOGLE FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* GLOBAL THEME */
    .stApp { 
        background-color: #0e1117; /* Deep Obsidian */
        font-family: 'Inter', sans-serif;
    }

    /* TYPOGRAPHY - HIGH CONTRAST */
    h1, h2, h3 { 
        color: #f0f2f6 !important; 
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    h1 { 
        font-size: 2.5rem !important; 
        background: -webkit-linear-gradient(45deg, #c9a66b, #f5d79b); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        text-shadow: 0px 0px 20px rgba(201, 166, 107, 0.3);
    }
    h2 { font-size: 1.8rem !important; border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-top: 30px; }
    h3 { font-size: 1.4rem !important; color: #c9a66b !important; }
    
    /* GENERAL TEXT VISIBILITY */
    p, li, span, div {
        color: #e6edf3; /* Bright Silver-White */
    }

    /* LABELS (Input Titles) */
    label, .stTextInput label, .stNumberInput label {
        color: #ffffff !important; /* Pure White */
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px;
    }

    /* ALERT/INFO BOXES */
    div[data-testid="stAlert"] {
        background-color: rgba(20, 26, 35, 0.8) !important; 
        border: 1px solid #3b82f6 !important; 
        color: #ffffff !important; 
    }
    div[data-testid="stAlert"] p {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { 
        background-color: #161b22; 
        border-right: 1px solid #2d333b;
    }

    /* METRIC CARDS */
    div[data-testid="stMetricValue"] { 
        color: #c9a66b; /* Muted Gold */
        font-size: 32px; 
        font-weight: 700;
        text-shadow: 0px 0px 10px rgba(201, 166, 107, 0.2);
    }
    div[data-testid="stMetricLabel"] { 
        color: #8b949e; 
        font-size: 14px; 
        font-weight: 600; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* INPUT FIELDS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input { 
        color: #ffffff !important; /* White Text Input */
        background-color: #0d1117 !important; 
        border: 1px solid #30363d !important; 
        border-radius: 6px;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #c9a66b !important;
        box-shadow: 0 0 0 1px #c9a66b !important;
    }
    
    /* BUTTONS */
    .stButton>button {
        background: linear-gradient(135deg, #c9a66b 0%, #a6854e 100%);
        color: #000000 !important; /* Black text on Gold */
        border: none;
        border-radius: 8px;
        font-weight: 800;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(201, 166, 107, 0.3);
    }
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(201, 166, 107, 0.5);
        color: #000000 !important;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #8b949e; 
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(201, 166, 107, 0.1);
        color: #c9a66b; 
        border-bottom: 2px solid #c9a66b;
    }
    
    /* EXPANDERS */
    .streamlit-expanderHeader {
        background-color: #161b22;
        border-radius: 6px;
        color: #e6edf3;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---

def format_large(num):
    if num is None: return "N/A"
    if abs(num) >= 1e12: return f"{num/1e12:.2f}T"
    if abs(num) >= 1e9: return f"{num/1e9:.2f}B"
    if abs(num) >= 1e6: return f"{num/1e6:.2f}M"
    return f"{num:,.0f}"

@st.cache_data
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Price History (Timezone Fix Applied)
        hist = stock.history(period="2y")
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)

        try:
            rf_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        except:
            rf_rate = 0.045

        bs = stock.balance_sheet
        inc = stock.financials
        
        if bs.empty or inc.empty: return None

        def get_val(df, keys):
            for k in keys:
                if k in df.index: return df.loc[k].iloc[0]
            return 0

        total_assets = get_val(bs, ['Total Assets', 'Assets'])
        total_liab = get_val(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
        retained_earnings = get_val(bs, ['Retained Earnings'])
        curr_assets = get_val(bs, ['Current Assets'])
        curr_liab = get_val(bs, ['Current Liabilities'])
        
        total_debt = get_val(bs, ['Total Debt'])
        if total_debt == 0: total_debt = info.get('totalDebt', 0)
        
        cash = get_val(bs, ['Cash And Cash Equivalents', 'Cash'])
        
        revenue = get_val(inc, ['Total Revenue', 'Revenue'])
        ebit = get_val(inc, ['EBIT', 'Operating Income'])
        interest_expense = abs(get_val(inc, ['Interest Expense']))
        tax_provision = get_val(inc, ['Tax Provision', 'Income Tax Expense'])
        pretax_income = get_val(inc, ['Pretax Income'])
        
        if pretax_income != 0 and tax_provision != 0:
            eff_tax_rate = tax_provision / pretax_income
            eff_tax_rate = max(0.0, min(eff_tax_rate, 0.40))
        else:
            eff_tax_rate = 0.25 
            
        if total_debt > 0 and interest_expense > 0:
            calc_cost_debt = interest_expense / total_debt
            calc_cost_debt = max(0.01, min(calc_cost_debt, 0.15))
        else:
            calc_cost_debt = 0.055

        return {
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
            "calc_cost_debt": calc_cost_debt
        }
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

def process_uploaded_data(df_upload):
    try:
        row = df_upload.iloc[0]
        dates = pd.date_range(end=datetime.now(), periods=100)
        dummy_hist = pd.DataFrame({'Close': np.linspace(row['Price'] * 0.8, row['Price'], 100),
                                   'Open': np.linspace(row['Price'] * 0.8, row['Price'], 100),
                                   'High': np.linspace(row['Price'] * 0.8, row['Price'], 100) * 1.05,
                                   'Low': np.linspace(row['Price'] * 0.8, row['Price'], 100) * 0.95}, index=dates)
        
        return {
            "name": str(row.get('Company Name', 'Custom Upload')),
            "sector": str(row.get('Sector', 'Unknown')),
            "summary": "Data provided via manual upload.",
            "currency": "USD",
            "price": float(row.get('Price', 0)),
            "mkt_cap": float(row.get('Market Cap', 0)),
            "beta": float(row.get('Beta', 1.0)),
            "shares": float(row.get('Shares Outstanding', 1000000)),
            "rf_rate": 0.045,
            "hist": dummy_hist,
            "revenue": float(row.get('Revenue', 0)),
            "ebit": float(row.get('EBIT', 0)),
            "total_debt": float(row.get('Total Debt', 0)),
            "cash": float(row.get('Cash', 0)),
            "total_assets": float(row.get('Total Assets', 0)),
            "total_liab": float(row.get('Total Liabilities', 0)),
            "retained_earnings": float(row.get('Retained Earnings', 0)),
            "wc": float(row.get('Working Capital', 0)),
            "eff_tax_rate": 0.25,
            "calc_cost_debt": 0.05
        }
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

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

def calculate_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# --- 3. APP LAYOUT ---

with st.sidebar:
    st.title("GT Terminal")
    st.markdown("`PROFESSIONAL SUITE V12.5`")
    st.markdown("---")
    nav = st.radio("NAVIGATION", [
        "Project Setup", 
        "Live Market Data", 
        "Valuation Model (DCF)", 
        "Peer Analysis", 
        "Risk Simulation", 
        "Financial Health"
    ])
    st.markdown("---")
    
    st.subheader("DATA ACTIONS")
    if 'data' in st.session_state and 'results' in st.session_state:
        d = st.session_state['data']
        res = st.session_state['results']
        
        report_text = f"""
        GT VALUATION TERMINAL - EXECUTIVE SUMMARY
        -----------------------------------------
        Company: {d['name']}
        Date: {datetime.now().strftime('%Y-%m-%d')}
        Sector: {d['sector']}
        
        VALUATION RESULTS
        -----------------
        Bear Case: ${res['Bear']:.2f}
        Base Case: ${res['Base']:.2f}
        Bull Case: ${res['Bull']:.2f}
        
        Upside (Base): {((res['Base'] / d['price'] if d['price'] > 0 else 0) - 1) * 100:.1f}%
        """
        
        st.download_button("📄 Download Report", report_text, f"{d['name']}_Report.txt")
        
        try:
            import xlsxwriter
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                d['hist'].to_excel(writer, sheet_name='Price_History')
                pd.DataFrame([d]).astype(str).to_excel(writer, sheet_name='Fundamentals')
                if 'scenarios' in st.session_state:
                    for k, v in st.session_state['scenarios'].items():
                        v.to_excel(writer, sheet_name=f'{k}_Case')
            st.download_button("📊 Export Model", output.getvalue(), f"{d['name']}_Data.xlsx")
        except:
            st.caption("Install xlsxwriter for Excel export.")
        
    elif 'data' in st.session_state:
        st.caption("Run Valuation to enable export.")
    else:
        st.caption("No active project.")

# ----------------------------
# 1. PROJECT SETUP
# ----------------------------
if nav == "Project Setup":
    st.title("Project Initialization")
    
    tab1, tab2 = st.tabs(["LIVE SEARCH", "MANUAL UPLOAD"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("Input ticker to fetch real-time market data.")
            ticker_input = st.text_input("Ticker Symbol", "AAPL").upper()
            
        if st.button("Initialize Live Model"):
            with st.spinner(f"Connecting to Exchange for {ticker_input}..."):
                data = get_financial_data(ticker_input)
                if data:
                    st.session_state['data'] = data
                    st.session_state['ticker'] = ticker_input
                    
                    base_rev = data['revenue']
                    year = datetime.now().year
                    scenarios = {
                        'Bear': project_scenario(base_rev, 5, 0.02, 0.20, year),
                        'Base': project_scenario(base_rev, 5, 0.08, 0.30, year),
                        'Bull': project_scenario(base_rev, 5, 0.15, 0.35, year)
                    }
                    st.session_state['scenarios'] = scenarios
                    st.success(f"Data Loaded: {data['name']}")
                else:
                    st.error("Ticker not found.")

    with tab2:
        st.info("Upload private company data (CSV).")
        template_data = {
            'Company Name': ['My Company'], 'Sector': ['Tech'], 'Price': [50], 
            'Market Cap': [1e8], 'Revenue': [2e7], 'EBIT': [5e6], 
            'Total Assets': [4e7], 'Total Liabilities': [1.5e7], 
            'Total Debt': [5e6], 'Cash': [2e6], 
            'Retained Earnings': [1e7], 'Working Capital': [5e6], 
            'Shares Outstanding': [2e6], 'Beta': [1.2]
        }
        df_temp = pd.DataFrame(template_data)
        csv = df_temp.to_csv(index=False).encode('utf-8')
        st.download_button("Download Template", csv, "template.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None and st.button("Initialize Manual Model"):
            try:
                df_up = pd.read_csv(uploaded_file)
                data = process_uploaded_data(df_up)
                if data:
                    st.session_state['data'] = data
                    st.session_state['ticker'] = "CUSTOM"
                    base_rev = data['revenue']
                    year = datetime.now().year
                    scenarios = {
                        'Bear': project_scenario(base_rev, 5, 0.02, 0.20, year),
                        'Base': project_scenario(base_rev, 5, 0.08, 0.30, year),
                        'Bull': project_scenario(base_rev, 5, 0.15, 0.35, year)
                    }
                    st.session_state['scenarios'] = scenarios
                    st.success(f"Custom Data Loaded: {data['name']}")
            except Exception as e:
                st.error(f"File Error: {e}")

    if 'scenarios' in st.session_state:
        st.markdown("### Scenario Visualization (3D)")
        combined = pd.DataFrame()
        for k, v in st.session_state['scenarios'].items():
            temp = v.copy()
            temp['Case'] = k
            combined = pd.concat([combined, temp])
        
        fig = px.scatter_3d(combined, x='Year', y='Case', z='Revenue',
                            color='Case', size='Implied_EBITDA',
                            color_discrete_map={'Bear':'#ef553b', 'Base':'#f1c40f', 'Bull':'#00cc96'})
        fig.update_layout(scene = dict(
                            xaxis_title='YEAR', yaxis_title='CASE', zaxis_title='REVENUE'),
                          margin=dict(l=0, r=0, b=0, t=0),
                          height=500, paper_bgcolor='rgba(0,0,0,0)', 
                          font=dict(color='#8b949e'))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 2. LIVE MARKET
# ----------------------------
elif nav == "Live Market Data":
    st.title("Market Terminal")
    
    if 'data' not in st.session_state:
        st.warning("Initialize project first.")
    else:
        d = st.session_state['data']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${d['price']:,.2f}")
        m2.metric("Market Cap", format_large(d['mkt_cap']))
        m3.metric("Beta (Volatility)", f"{d['beta']:.2f}")
        m4.metric("Risk-Free Rate", f"{d['rf_rate']:.2%}")
        
        st.subheader("Technical Analysis")
        if not d['hist'].empty:
            hist = calculate_technical_indicators(d['hist'].copy())
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                         low=hist['Low'], close=hist['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='#f1c40f', width=1.5), name='SMA 50'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='#3b82f6', width=1.5), name='SMA 200'))
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#8b949e'), xaxis_rangeslider_visible=False,
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#30363d'))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sentiment Analysis")
        try:
            if st.session_state.get('ticker') != "CUSTOM":
                # Create Ticker Object
                ticker_obj = yf.Ticker(st.session_state['ticker'])
                news = ticker_obj.news
                
                if news:
                    counter = 0
                    for n in news:
                        if counter >= 3: break
                        
                        title = n.get('title', 'No Title')
                        link = n.get('link', '#')
                        
                        # Perform Sentiment Analysis
                        blob = TextBlob(title)
                        score = blob.sentiment.polarity
                        
                        if score > 0.1: color = "#00cc96" 
                        elif score < -0.1: color = "#ef553b" 
                        else: color = "#8b949e" 
                            
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(22, 27, 34, 0.5); padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {color};">
                                <a href="{link}" target="_blank" style="color: #e6edf3; text-decoration: none; font-weight: 600;">{title}</a>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        counter += 1
                else:
                    st.info("No recent news found for this ticker.")
            else:
                st.info("News disabled for custom data.")
        except Exception as e:
            st.error(f"News Error: {e}")

# ----------------------------
# 3. VALUATION (DCF)
# ----------------------------
elif nav == "Valuation Model (DCF)":
    st.title("Intrinsic Valuation (DCF)")
    
    if 'data' not in st.session_state:
        st.warning("Initialize project first.")
    else:
        d = st.session_state['data']
        
        with st.expander("⚙️ MODEL ASSUMPTIONS", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                cost_equity = st.number_input("Cost of Equity (Ke) %", value=0.10, step=0.005, format="%.3f")
            with c2:
                cost_debt = st.number_input("Cost of Debt (Kd) %", value=d['calc_cost_debt'], step=0.005, format="%.3f")
            with c3:
                tax_rate = st.number_input("Tax Rate %", value=d['eff_tax_rate'], step=0.01, format="%.2f")
            
            c4, c5 = st.columns(2)
            with c4:
                debt_weight = st.slider("Capital Structure (Debt %)", 0, 100, 20) / 100
            with c5:
                tgr = st.slider("Terminal Growth Rate %", 1.0, 5.0, 2.5) / 100
                st.session_state['tgr'] = tgr
                exit_mult = st.number_input("Exit Multiple (EV/EBITDA)", value=12.0)

        wacc = (cost_equity * (1 - debt_weight)) + (cost_debt * (1 - tax_rate) * debt_weight)
        st.session_state['wacc'] = wacc
        st.metric("WACC", f"{wacc:.2%}", delta="Weighted Avg Cost of Capital", delta_color="off")
        
        results = {}
        for name, df in st.session_state['scenarios'].items():
            df['FCF_Proxy'] = df['Implied_EBITDA'] * (1 - tax_rate) * 0.7
            df['Discount_Factor'] = [(1 + wacc) ** i for i in range(1, 6)]
            df['PV'] = df['FCF_Proxy'] / df['Discount_Factor']
            pv_explicit = df['PV'].sum()
            
            last_fcf = df['FCF_Proxy'].iloc[-1]
            tv_gg = (last_fcf * (1 + tgr)) / (wacc - tgr)
            pv_tv_gg = tv_gg / ((1 + wacc) ** 5)
            
            last_ebitda = df['Implied_EBITDA'].iloc[-1]
            tv_em = last_ebitda * exit_mult
            pv_tv_em = tv_em / ((1 + wacc) ** 5)
            
            ev = pv_explicit + ((pv_tv_gg + pv_tv_em) / 2)
            net_debt = d['total_debt'] - d['cash']
            equity_val = ev - net_debt
            share_price = equity_val / d['shares']
            results[name] = share_price
        
        st.session_state['results'] = results

        st.subheader("Implied Share Price")
        c1, c2, c3 = st.columns(3)
        c1.metric("BEAR CASE", f"${results['Bear']:.2f}")
        c2.metric("BASE CASE", f"${results['Base']:.2f}")
        c3.metric("BULL CASE", f"${results['Bull']:.2f}")
        
        fig = go.Figure()
        for k, v in results.items():
            color = {'Bear':'#ef553b','Base':'#f1c40f','Bull':'#00cc96'}[k]
            fig.add_trace(go.Bar(y=[k], x=[v], orientation='h', text=f"${v:.2f}", 
                                 marker=dict(color=color, line=dict(width=0)), opacity=0.9))
        fig.update_layout(title="Valuation Range", height=300, paper_bgcolor='rgba(0,0,0,0)', 
                          plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sensitivity Surface (WACC vs Growth)")
        wacc_range = np.linspace(wacc-0.02, wacc+0.02, 10)
        tgr_range = np.linspace(tgr-0.01, tgr+0.01, 10)
        X, Y = np.meshgrid(wacc_range, tgr_range)
        Z = np.zeros_like(X)
        
        for i in range(len(tgr_range)):
            for j in range(len(wacc_range)):
                w = X[i, j]
                t = Y[i, j]
                tv = (last_fcf * (1 + t)) / (w - t)
                val = (pv_explicit + (tv / ((1+w)**5)) - net_debt) / d['shares']
                Z[i, j] = val

        fig_3d = go.Figure(data=[go.Surface(z=Z, x=wacc_range, y=tgr_range, colorscale='Viridis')])
        fig_3d.update_layout(scene = dict(xaxis_title='WACC', yaxis_title='GROWTH', zaxis_title='PRICE'),
                             height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'),
                             margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_3d, use_container_width=True)

# ----------------------------
# 4. PEER ANALYSIS
# ----------------------------
elif nav == "Peer Analysis":
    st.title("Relative Valuation")
    peers = st.text_input("Peer Group (Comma Separated)", "GOOG, MSFT, META, AMZN").upper()
    
    if st.button("Generate Comparison"):
        tickers = [t.strip() for t in peers.split(",")]
        rows = []
        with st.spinner("Aggregating peer data..."):
            for t in tickers:
                try:
                    i = yf.Ticker(t).info
                    rows.append({
                        "Ticker": t,
                        "Price": i.get('currentPrice'),
                        "P/E": i.get('trailingPE'),
                        "EV/EBITDA": i.get('enterpriseToEbitda'),
                        "P/B": i.get('priceToBook'),
                        "ROE": i.get('returnOnEquity', 0) * 100 if i.get('returnOnEquity') else None
                    })
                except: pass
        
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({"Price": "${:.2f}", "P/E": "{:.1f}x", "EV/EBITDA": "{:.1f}x", "ROE": "{:.1f}%"}))
            
            fig = px.scatter(df, x="P/E", y="EV/EBITDA", text="Ticker", size="Price", 
                             color="Ticker", title="Multiples Mapping")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#8b949e'), xaxis=dict(gridcolor='#30363d'), 
                              yaxis=dict(gridcolor='#30363d'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data available.")

# ----------------------------
# 5. RISK SIMULATION
# ----------------------------
elif nav == "Risk Simulation":
    st.title("Monte Carlo Simulation")
    if 'scenarios' not in st.session_state:
        st.warning("Initialize project first.")
    else:
        base_price = st.session_state['results']['Base'] if 'results' in st.session_state else 150.0
            
        volatility = st.slider("Implied Volatility (σ)", 10, 50, 25) / 100
        sims = st.selectbox("Simulation Iterations", [1000, 5000, 10000])
        
        if st.button("Run Monte Carlo"):
            results = base_price * np.exp(np.random.normal(-0.5 * volatility**2, volatility, sims))
            
            fig = px.histogram(results, nbins=50, title="Probability Distribution (1 Year)", 
                               color_discrete_sequence=['#c9a66b'])
            fig.add_vline(x=base_price, line_dash="dash", line_color="white", annotation_text="Base Case")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#8b949e'), bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Value at Risk (95%)", f"${np.percentile(results, 5):.2f}", delta="-5% Risk Floor", delta_color="inverse")

# ----------------------------
# 6. FINANCIAL HEALTH
# ----------------------------
elif nav == "Financial Health":
    st.title("Health Diagnostics")
    
    if 'data' not in st.session_state:
        st.warning("Initialize project first.")
    else:
        d = st.session_state['data']
        is_bank = 'Financial' in d['sector'] or 'Bank' in d['name']
        
        if is_bank:
            st.warning(f"Financial Sector: Z-Score not applicable for {d['name']}.")
        else:
            try:
                ta = d['total_assets'] if d['total_assets'] > 0 else 1
                tl = d['total_liab'] if d['total_liab'] > 0 else 1
                
                A = d['wc'] / ta
                B = d['retained_earnings'] / ta
                C = d['ebit'] / ta
                D = d['mkt_cap'] / tl
                E = d['revenue'] / ta
                
                z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
                
                st.metric("Altman Z-Score", f"{z:.2f}")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=z,
                    gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "#c9a66b"},
                           'steps': [{'range': [0, 1.8], 'color': "#ef553b"},
                                     {'range': [1.8, 3], 'color': "#f1c40f"},
                                     {'range': [3, 5], 'color': "#00cc96"}]}
                ))
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'))
                st.plotly_chart(fig, use_container_width=True)
                
                if z > 3: st.success("Condition: STABLE")
                elif z > 1.8: st.warning("Condition: WATCHLIST")
                else: st.error("Condition: DISTRESSED")
            except:
                st.error("Insufficient Data.")
