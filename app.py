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
st.set_page_config(page_title="GT Valuation Terminal", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    /* Main Theme Colors */
    .stApp { background-color: #1e1e1e; }
    
    /* Typography */
    h1, h2, h3 { color: #d4af37 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    p, li, span { color: #e0e0e0; }
    label { color: #ffffff !important; font-weight: 600; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid #333; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #d4af37; font-size: 28px; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #a0a0a0; font-size: 14px; }
    
    /* Inputs */
    .stTextInput>div>div>input { color: black !important; background-color: #f0f2f6 !important; font-weight: bold; }
    .stNumberInput>div>div>input { color: black !important; background-color: #f0f2f6 !important; font-weight: bold; }
    
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

# --- 2. DATA ENGINE ---

def format_large(num):
    if num is None: return "N/A"
    if abs(num) >= 1e12: return f"{num/1e12:.2f}T"
    if abs(num) >= 1e9: return f"{num/1e9:.2f}B"
    if abs(num) >= 1e6: return f"{num/1e6:.2f}M"
    return f"{num:,.0f}"

@st.cache_data
def get_financial_data(ticker):
    """
    Robust data fetcher that handles Banks/Non-Banks and calculates derived metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Price History
        hist = stock.history(period="2y")
        try:
            rf_rate = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1] / 100
        except:
            rf_rate = 0.045

        # 2. Financial Statements
        bs = stock.balance_sheet
        inc = stock.financials
        cf = stock.cashflow
        
        if bs.empty or inc.empty: return None

        # 3. Smart Metric Extraction (Handles missing keys)
        def get_val(df, keys):
            for k in keys:
                if k in df.index: return df.loc[k].iloc[0]
            return 0

        # Balance Sheet
        total_assets = get_val(bs, ['Total Assets', 'Assets'])
        total_liab = get_val(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
        retained_earnings = get_val(bs, ['Retained Earnings'])
        curr_assets = get_val(bs, ['Current Assets'])
        curr_liab = get_val(bs, ['Current Liabilities'])
        
        total_debt = get_val(bs, ['Total Debt'])
        if total_debt == 0: total_debt = info.get('totalDebt', 0)
        
        cash = get_val(bs, ['Cash And Cash Equivalents', 'Cash'])
        
        # Income Statement
        revenue = get_val(inc, ['Total Revenue', 'Revenue'])
        ebit = get_val(inc, ['EBIT', 'Operating Income'])
        interest_expense = abs(get_val(inc, ['Interest Expense']))
        tax_provision = get_val(inc, ['Tax Provision', 'Income Tax Expense'])
        pretax_income = get_val(inc, ['Pretax Income'])
        
        # 4. Calculated Defaults (Smart Inputs)
        # Tax Rate
        if pretax_income != 0 and tax_provision != 0:
            eff_tax_rate = tax_provision / pretax_income
            eff_tax_rate = max(0.0, min(eff_tax_rate, 0.40))
        else:
            eff_tax_rate = 0.25 
            
        # Cost of Debt
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
    """Parses user uploaded CSV into the app's dictionary format"""
    try:
        # Expecting a single row CSV with specific headers
        row = df_upload.iloc[0]
        # Create dummy history for chart compatibility
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
            "rf_rate": 0.045, # Default
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

# SIDEBAR NAV
with st.sidebar:
    st.title("GT Terminal 🚀")
    st.caption("Professional Edition v12.0 (3D Enabled)")
    st.markdown("---")
    nav = st.radio("Modules", [
        "Project Setup", 
        "Live Market Data", 
        "Valuation Model (DCF)", 
        "Peer Analysis", 
        "Risk Simulation", 
        "Financial Health"
    ])
    st.markdown("---")
    
    # --- NEW FEATURE: REPORT GENERATOR ---
    st.subheader("🖨️ Report Center")
    if 'data' in st.session_state and 'results' in st.session_state:
        d = st.session_state['data']
        res = st.session_state['results']
        
        # Text Report
        report_text = f"""
        GT VALUATION TERMINAL - EXECUTIVE SUMMARY
        -----------------------------------------
        Company: {d['name']}
        Date: {datetime.now().strftime('%Y-%m-%d')}
        Sector: {d['sector']}
        
        VALUATION RESULTS (Intrinsic Value per Share)
        -----------------------------------------
        Bear Case: ${res['Bear']:.2f}
        Base Case: ${res['Base']:.2f}
        Bull Case: ${res['Bull']:.2f}
        
        Current Market Price: ${d['price']:.2f}
        Upside (Base): {((res['Base'] / d['price'] if d['price'] > 0 else 0) - 1) * 100:.1f}%
        
        KEY INPUTS
        -----------------------------------------
        WACC: {st.session_state.get('wacc', 0):.2%}
        Terminal Growth: {st.session_state.get('tgr', 0):.2%}
        Net Debt: ${format_large(d['total_debt'] - d['cash'])}
        """
        
        st.download_button("📄 Download Summary (.txt)", report_text, f"{d['name']}_Report.txt")
        
        # Data Dump
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            d['hist'].to_excel(writer, sheet_name='Price_History')
            pd.DataFrame([d]).astype(str).to_excel(writer, sheet_name='Fundamentals')
            if 'scenarios' in st.session_state:
                for k, v in st.session_state['scenarios'].items():
                    v.to_excel(writer, sheet_name=f'{k}_Case')
        
        st.download_button("📊 Download Data (.xlsx)", output.getvalue(), f"{d['name']}_Data.xlsx")
        
    elif 'data' in st.session_state:
        st.caption("Run Valuation to enable full report.")
    else:
        st.caption("Initialize project to enable reports.")


# ----------------------------
# 1. PROJECT SETUP (ENHANCED)
# ----------------------------
if nav == "Project Setup":
    st.title("📂 Project Initialization")
    
    # Tabs for Automatic vs Manual
    tab1, tab2 = st.tabs(["🌎 Live Ticker Search", "📤 Upload Manual Data"])
    
    # MODE A: Live Search
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("Auto-fetch financials and build scenarios.")
            ticker_input = st.text_input("Ticker Symbol", "AAPL").upper()
            
        if st.button("🚀 Initialize Model (Live)"):
            with st.spinner(f"Analyzing {ticker_input}..."):
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
                    st.success(f"Successfully loaded data for {data['name']}")
                else:
                    st.error("Could not fetch data.")

    # MODE B: Manual Upload
    with tab2:
        st.info("Upload a CSV for private companies or custom data.")
        
        # Template Generator
        template_data = {
            'Company Name': ['My Company Ltd'], 'Sector': ['Technology'], 'Price': [50], 
            'Market Cap': [100000000], 'Revenue': [20000000], 'EBIT': [5000000], 
            'Total Assets': [40000000], 'Total Liabilities': [15000000], 
            'Total Debt': [5000000], 'Cash': [2000000], 
            'Retained Earnings': [10000000], 'Working Capital': [5000000], 
            'Shares Outstanding': [2000000], 'Beta': [1.2]
        }
        df_temp = pd.DataFrame(template_data)
        csv = df_temp.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Template", csv, "template.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Upload Completed CSV", type=['csv'])
        
        if uploaded_file is not None:
            if st.button("🚀 Initialize Model (Manual)"):
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
                        st.success(f"Successfully loaded custom data for {data['name']}")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    # Visualization
    if 'scenarios' in st.session_state:
        st.markdown("### 📊 Scenario Preview (3D Analysis)")
        combined = pd.DataFrame()
        for k, v in st.session_state['scenarios'].items():
            temp = v.copy()
            temp['Case'] = k
            combined = pd.concat([combined, temp])
        
        # NEW: 3D Visualization of Scenarios
        fig = px.scatter_3d(combined, x='Year', y='Case', z='Revenue',
                            color='Case', size='Implied_EBITDA',
                            color_discrete_map={'Bear':'#ef553b', 'Base':'#f1c40f', 'Bull':'#00cc96'})
        fig.update_layout(scene = dict(
                            xaxis_title='Year',
                            yaxis_title='Scenario',
                            zaxis_title='Revenue ($)'),
                          margin=dict(l=0, r=0, b=0, t=0),
                          height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 2. LIVE MARKET
# ----------------------------
elif nav == "Live Market Data":
    st.title("📈 Market Terminal")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize a ticker in 'Project Setup' first.")
    else:
        d = st.session_state['data']
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"${d['price']:,.2f}")
        m2.metric("Market Cap", format_large(d['mkt_cap']))
        m3.metric("Beta", f"{d['beta']:.2f}")
        m4.metric("Risk-Free Rate", f"{d['rf_rate']:.2%}")
        
        # Technical Chart
        st.subheader("Price Action & Technicals")
        if not d['hist'].empty:
            hist = calculate_technical_indicators(d['hist'].copy())
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                         low=hist['Low'], close=hist['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='#f1c40f'), name='SMA 50'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='#00cc96'), name='SMA 200'))
            fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#e0e0e0'), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical data not available for this dataset.")
        
        # News Sentiment
        st.subheader("📰 News Sentiment")
        try:
            if st.session_state.get('ticker') != "CUSTOM":
                news = yf.Ticker(st.session_state['ticker']).news
                if news:
                    for n in news[:3]:
                        blob = TextBlob(n['title'])
                        score = blob.sentiment.polarity
                        color = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                        st.markdown(f"{color} [{n['title']}]({n['link']})")
                else:
                    st.info("No recent news found.")
            else:
                st.info("News feed disabled for Custom Uploads.")
        except:
            st.info("News feed unavailable.")

# ----------------------------
# 3. VALUATION (DCF) (ENHANCED)
# ----------------------------
elif nav == "Valuation Model (DCF)":
    st.title("💎 Intrinsic Valuation")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize a ticker in 'Project Setup' first.")
    else:
        d = st.session_state['data']
        
        # --- INPUTS ---
        with st.expander("⚙️ Model Assumptions (Smart Defaults Applied)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                cost_equity = st.number_input("Cost of Equity (%)", value=0.10, step=0.005, format="%.3f")
            with c2:
                cost_debt = st.number_input("Cost of Debt (%)", value=d['calc_cost_debt'], step=0.005, format="%.3f")
            with c3:
                tax_rate = st.number_input("Tax Rate (%)", value=d['eff_tax_rate'], step=0.01, format="%.2f")
            
            c4, c5 = st.columns(2)
            with c4:
                debt_weight = st.slider("Debt Weight %", 0, 100, 20) / 100
            with c5:
                tgr = st.slider("Terminal Growth Rate %", 1.0, 5.0, 2.5) / 100
                st.session_state['tgr'] = tgr
                exit_mult = st.number_input("Exit Multiple (EV/EBITDA)", value=12.0)

        # WACC Calc
        wacc = (cost_equity * (1 - debt_weight)) + (cost_debt * (1 - tax_rate) * debt_weight)
        st.session_state['wacc'] = wacc
        st.metric("Calculated WACC", f"{wacc:.2%}")
        
        # --- DCF ENGINE ---
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

        # --- OUTPUTS ---
        st.subheader("🎯 Implied Share Price (Intrinsic Value)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bear Case", f"${results['Bear']:.2f}")
        c2.metric("Base Case", f"${results['Base']:.2f}")
        c3.metric("Bull Case", f"${results['Bull']:.2f}")
        
        # Football Field
        fig = go.Figure()
        for k, v in results.items():
            fig.add_trace(go.Bar(y=[k], x=[v], orientation='h', text=f"${v:.2f}", 
                                 marker_color={'Bear':'#ef553b','Base':'#f1c40f','Bull':'#00cc96'}[k]))
        fig.update_layout(title="Valuation Range", height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)
        
        # --- NEW: 3D SENSITIVITY MATRIX ---
        st.subheader("Sensitivity Analysis (3D Surface)")
        st.caption("Visualizing Valuation impact based on WACC (X) and Terminal Growth (Y)")
        
        wacc_range = np.linspace(wacc-0.02, wacc+0.02, 10)
        tgr_range = np.linspace(tgr-0.01, tgr+0.01, 10)
        
        # Create Meshgrid for 3D
        X, Y = np.meshgrid(wacc_range, tgr_range)
        Z = np.zeros_like(X)
        
        for i in range(len(tgr_range)):
            for j in range(len(wacc_range)):
                w = X[i, j]
                t = Y[i, j]
                # Recalc TV
                tv = (last_fcf * (1 + t)) / (w - t)
                val = (pv_explicit + (tv / ((1+w)**5)) - net_debt) / d['shares']
                Z[i, j] = val

        # 3D Surface Plot
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=wacc_range, y=tgr_range, colorscale='Viridis')])
        fig_3d.update_layout(title='Valuation Surface', autosize=True,
                             scene = dict(
                                xaxis_title='WACC',
                                yaxis_title='Growth Rate',
                                zaxis_title='Share Price'),
                             width=800, height=500,
                             paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig_3d, use_container_width=True)

# ----------------------------
# 4. PEER ANALYSIS
# ----------------------------
elif nav == "Peer Analysis":
    st.title("🌍 Relative Valuation")
    peers = st.text_input("Enter Peer Tickers (comma separated)", "GOOG, MSFT, META, AMZN").upper()
    
    if st.button("Run Comparison"):
        tickers = [t.strip() for t in peers.split(",")]
        rows = []
        with st.spinner("Fetching peer data..."):
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
            
            fig = px.scatter(df, x="P/E", y="EV/EBITDA", text="Ticker", size="Price", color="Ticker", title="Peer Mapping")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data found for peers.")

# ----------------------------
# 5. RISK SIMULATION
# ----------------------------
elif nav == "Risk Simulation":
    st.title("🎲 Monte Carlo Risk Assessment")
    if 'scenarios' not in st.session_state:
        st.warning("Please build model first.")
    else:
        if 'results' in st.session_state:
            base_price = st.session_state['results']['Base']
        else:
            base_price = 150.0
            
        volatility = st.slider("Implied Volatility (%)", 10, 50, 25) / 100
        sims = st.selectbox("Iterations", [1000, 5000, 10000])
        
        if st.button("Run Simulation"):
            results = base_price * np.exp(np.random.normal(-0.5 * volatility**2, volatility, sims))
            
            fig = px.histogram(results, nbins=50, title="Projected Price Distribution (1 Year)", 
                               color_discrete_sequence=['#d4af37'])
            fig.add_vline(x=base_price, line_dash="dash", line_color="white", annotation_text="Base")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("VaR (95% Confidence)", f"${np.percentile(results, 5):.2f}")

# ----------------------------
# 6. FINANCIAL HEALTH
# ----------------------------
elif nav == "Financial Health":
    st.title("🏥 Financial Health Check")
    
    if 'data' not in st.session_state:
        st.warning("Please initialize a ticker first.")
    else:
        d = st.session_state['data']
        
        is_bank = 'Financial' in d['sector'] or 'Bank' in d['name']
        
        if is_bank:
            st.warning(f"⚠️ **Sector Notice:** {d['name']} is identified as a Financial Institution.")
            st.info("Traditional Z-Scores are invalid for banks due to high deposit liabilities. "
                    "Focus on Capital Adequacy and Liquidity instead.")
        else:
            try:
                # Avoid division by zero
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
                    gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "white"},
                           'steps': [{'range': [0, 1.8], 'color': "#ef553b"},
                                     {'range': [1.8, 3], 'color': "#f1c40f"},
                                     {'range': [3, 5], 'color': "#00cc96"}]}
                ))
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
                st.plotly_chart(fig, use_container_width=True)
                
                if z > 3: st.success("Safe Zone: Low Probability of Bankruptcy")
                elif z > 1.8: st.warning("Grey Zone: Caution Recommended")
                else: st.error("Distress Zone: High Risk")
                
            except:
                st.error("Insufficient data to calculate Z-Score.")
