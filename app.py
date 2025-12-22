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
            # Cap realistic bounds (0% to 40%)
            eff_tax_rate = max(0.0, min(eff_tax_rate, 0.40))
        else:
            eff_tax_rate = 0.25 # Default
            
        # Cost of Debt (Interest / Total Debt)
        if total_debt > 0 and interest_expense > 0:
            calc_cost_debt = interest_expense / total_debt
            calc_cost_debt = max(0.01, min(calc_cost_debt, 0.15)) # Cap between 1% and 15%
        else:
            calc_cost_debt = 0.055

        return {
            # Profile
            "name": info.get('longName', ticker),
            "sector": info.get('sector', 'Unknown'),
            "summary": info.get('longBusinessSummary', 'No description.'),
            "currency": info.get('currency', 'USD'),
            
            # Market
            "price": info.get('currentPrice', 0),
            "mkt_cap": info.get('marketCap', 0),
            "beta": info.get('beta', 1.0),
            "shares": info.get('sharesOutstanding', 1),
            "rf_rate": rf_rate,
            "hist": hist,
            
            # Fundamentals
            "revenue": revenue,
            "ebit": ebit,
            "total_debt": total_debt,
            "cash": cash,
            "total_assets": total_assets,
            "total_liab": total_liab,
            "retained_earnings": retained_earnings,
            "wc": curr_assets - curr_liab,
            
            # Smart Inputs
            "eff_tax_rate": eff_tax_rate,
            "calc_cost_debt": calc_cost_debt
        }
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

def project_scenario(base_rev, years, growth, margin, current_year):
    data = []
    curr_rev = base_rev
    for i in range(1, years + 1):
        curr_rev = curr_rev * (1 + growth)
        # Simplified FCF Proxy: EBITDA * (1-Tax) - Reinvestment
        # We assume Reinvestment (CapEx + NWC) scales with revenue
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
    st.caption("Professional Edition v11.0")
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

# ----------------------------
# 1. PROJECT SETUP
# ----------------------------
if nav == "Project Setup":
    st.title("📂 Project Initialization")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("Start by entering a ticker symbol. The model will auto-fetch financials and build 3 scenarios.")
        ticker_input = st.text_input("Ticker Symbol", "AAPL").upper()
        
    if st.button("🚀 Initialize Model"):
        with st.spinner(f"Analyzing {ticker_input}..."):
            data = get_financial_data(ticker_input)
            
            if data:
                st.session_state['data'] = data
                st.session_state['ticker'] = ticker_input
                
                # Auto-build Scenarios based on latest Revenue
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
                st.error("Could not fetch data. Please check ticker.")

    if 'scenarios' in st.session_state:
        st.markdown("### 📊 Scenario Preview")
        # Combine for chart
        combined = pd.DataFrame()
        for k, v in st.session_state['scenarios'].items():
            temp = v.copy()
            temp['Case'] = k
            combined = pd.concat([combined, temp])
            
        fig = px.bar(combined, x='Year', y='Revenue', color='Case', barmode='group',
                     color_discrete_map={'Bear':'#ef553b', 'Base':'#f1c40f', 'Bull':'#00cc96'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Export
        csv = combined.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Export Scenarios to CSV", csv, "model_scenarios.csv", "text/csv")

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
        hist = calculate_technical_indicators(d['hist'].copy())
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                     low=hist['Low'], close=hist['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='#f1c40f'), name='SMA 50'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='#00cc96'), name='SMA 200'))
        fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#e0e0e0'), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # News Sentiment
        st.subheader("📰 News Sentiment")
        try:
            news = yf.Ticker(st.session_state['ticker']).news
            if news:
                for n in news[:3]:
                    blob = TextBlob(n['title'])
                    score = blob.sentiment.polarity
                    color = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
                    st.markdown(f"{color} [{n['title']}]({n['link']})")
            else:
                st.info("No recent news found.")
        except:
            st.info("News feed unavailable.")

# ----------------------------
# 3. VALUATION (DCF)
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
            # Pre-filled with actual data where possible
            with c1:
                cost_equity = st.number_input("Cost of Equity (%)", value=0.10, step=0.005, format="%.3f")
            with c2:
                # Use smart default for cost of debt
                cost_debt = st.number_input("Cost of Debt (%)", value=d['calc_cost_debt'], step=0.005, format="%.3f")
            with c3:
                # Use smart default for tax rate
                tax_rate = st.number_input("Tax Rate (%)", value=d['eff_tax_rate'], step=0.01, format="%.2f")
            
            c4, c5 = st.columns(2)
            with c4:
                debt_weight = st.slider("Debt Weight %", 0, 100, 20) / 100
            with c5:
                tgr = st.slider("Terminal Growth Rate %", 1.0, 5.0, 2.5) / 100
                exit_mult = st.number_input("Exit Multiple (EV/EBITDA)", value=12.0)

        # WACC Calc
        wacc = (cost_equity * (1 - debt_weight)) + (cost_debt * (1 - tax_rate) * debt_weight)
        st.metric("Calculated WACC", f"{wacc:.2%}")
        
        # --- DCF ENGINE ---
        results = {}
        for name, df in st.session_state['scenarios'].items():
            # Proxy Free Cash Flow: EBITDA * (1 - Tax) * Conversion Ratio (approx 70%)
            # This is a robust simplification for general purpose
            df['FCF_Proxy'] = df['Implied_EBITDA'] * (1 - tax_rate) * 0.7
            
            # 1. PV of Explicit Period
            df['Discount_Factor'] = [(1 + wacc) ** i for i in range(1, 6)]
            df['PV'] = df['FCF_Proxy'] / df['Discount_Factor']
            pv_explicit = df['PV'].sum()
            
            # 2. Terminal Value (Method 1: Gordon Growth)
            last_fcf = df['FCF_Proxy'].iloc[-1]
            tv_gg = (last_fcf * (1 + tgr)) / (wacc - tgr)
            pv_tv_gg = tv_gg / ((1 + wacc) ** 5)
            
            # 3. Terminal Value (Method 2: Exit Multiple)
            last_ebitda = df['Implied_EBITDA'].iloc[-1]
            tv_em = last_ebitda * exit_mult
            pv_tv_em = tv_em / ((1 + wacc) ** 5)
            
            # 4. Enterprise Value (Average of both methods)
            ev = pv_explicit + ((pv_tv_gg + pv_tv_em) / 2)
            
            # 5. Equity Value
            net_debt = d['total_debt'] - d['cash']
            equity_val = ev - net_debt
            share_price = equity_val / d['shares']
            
            results[name] = share_price
            
            # Save Base Case DF for 3D
            if name == 'Base':
                base_df = df
                base_target = share_price

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
        
        # 2D Matrix (WACC vs Growth)
        st.subheader("Sensitivity Analysis (Base Case)")
        wacc_range = np.linspace(wacc-0.02, wacc+0.02, 5)
        tgr_range = np.linspace(tgr-0.01, tgr+0.01, 5)
        
        matrix = []
        for t in tgr_range:
            row = []
            for w in wacc_range:
                # Re-calc simple TV GG
                tv = (last_fcf * (1 + t)) / (w - t)
                val = (pv_explicit + (tv / ((1+w)**5)) - net_debt) / d['shares']
                row.append(val)
            matrix.append(row)
            
        df_mat = pd.DataFrame(matrix, index=[f"G: {x:.1%}" for x in tgr_range], 
                              columns=[f"W: {x:.1%}" for x in wacc_range])
        st.dataframe(df_mat.style.format("${:.2f}").background_gradient(cmap='RdYlGn', axis=None))

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
            
            # Scatter
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
        # Use Base Case Price from DCF logic (Simplified re-calc for demo)
        base_price = st.number_input("Target Price (Base Case)", value=150.0) # Placeholder, ideally linked
        volatility = st.slider("Implied Volatility (%)", 10, 50, 25) / 100
        sims = st.selectbox("Iterations", [1000, 5000, 10000])
        
        if st.button("Run Simulation"):
            # Random Walk
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
        
        # Check for Bank
        is_bank = 'Financial' in d['sector'] or 'Bank' in d['name']
        
        if is_bank:
            st.warning(f"⚠️ **Sector Notice:** {d['name']} is identified as a Financial Institution.")
            st.info("Traditional Z-Scores are invalid for banks due to high deposit liabilities. "
                    "Focus on Capital Adequacy and Liquidity instead.")
        else:
            # Z-Score Calc
            try:
                A = d['wc'] / d['total_assets']
                B = d['retained_earnings'] / d['total_assets']
                C = d['ebit'] / d['total_assets']
                D = d['mkt_cap'] / d['total_liab']
                E = d['revenue'] / d['total_assets']
                
                z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
                
                st.metric("Altman Z-Score", f"{z:.2f}")
                
                # Gauge
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
