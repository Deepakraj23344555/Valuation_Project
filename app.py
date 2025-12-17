# ==============================
# GT VALUATION TERMINAL (FINAL)
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime
from textblob import TextBlob
import base64

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="GT Valuation Terminal",
    page_icon="💼",
    layout="wide"
)

# --------------------------------------------------
# 2. GLOBAL SESSION STATE GUARD
# --------------------------------------------------
DEFAULT_STATE = {
    "scenarios": {},
    "ticker_data": {},
    "ticker_name": "",
    "ticker_symbol": "",
    "base_df": None,
    "ev_results": {},
    "wacc": 0.10,
    "comps": pd.DataFrame()
}

for k, v in DEFAULT_STATE.items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------
def safe_get(d, key, default=0):
    try:
        return d.get(key, default)
    except Exception:
        return default


@st.cache_data(ttl=3600)
def get_market_data(ticker: str):
    stock = yf.Ticker(ticker)

    try:
        hist = stock.history(period="2y")
        info = stock.fast_info
    except Exception as e:
        raise RuntimeError(f"Market data unavailable: {e}")

    try:
        rf_rate = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] / 100
    except Exception:
        rf_rate = 0.045

    bs = stock.balance_sheet
    fin = stock.financials

    def bs_val(k): return bs.loc[k].iloc[0] if k in bs.index else 0
    def fin_val(k): return fin.loc[k].iloc[0] if k in fin.index else 0

    return {
        "price": info.get("last_price", 0),
        "mkt_cap": info.get("market_cap", 0),
        "beta": info.get("beta", 1),
        "currency": info.get("currency", "USD"),
        "history": hist,
        "rf_rate": rf_rate,
        "sector": safe_get(stock.info, "sector", "Unknown"),
        "name": safe_get(stock.info, "longName", ticker),
        "summary": safe_get(stock.info, "longBusinessSummary", ""),
        "total_assets": bs_val("Total Assets"),
        "total_liab": bs_val("Total Liabilities Net Minority Interest"),
        "retained_earnings": bs_val("Retained Earnings"),
        "working_capital": bs_val("Current Assets") - bs_val("Current Liabilities"),
        "ebit": fin_val("EBIT"),
        "revenue": fin_val("Total Revenue"),
        "debt": safe_get(stock.info, "totalDebt", 0),
        "cash": safe_get(stock.info, "totalCash", 0),
        "shares": safe_get(stock.info, "sharesOutstanding", 1)
    }


def get_historical_profitability(ticker: str):
    stock = yf.Ticker(ticker)
    fin = stock.financials.T

    if fin.empty:
        return pd.DataFrame()

    revenue_col = next((c for c in ["Total Revenue", "Revenue"] if c in fin.columns), None)
    if revenue_col is None:
        return pd.DataFrame()

    df = pd.DataFrame(index=fin.index)
    df["Revenue"] = fin[revenue_col]

    if "Gross Profit" in fin.columns:
        df["Gross Margin"] = fin["Gross Profit"] / df["Revenue"] * 100

    op_col = next((c for c in ["Operating Income", "EBIT", "Pretax Income"] if c in fin.columns), None)
    if op_col:
        df["Operating Margin"] = fin[op_col] / df["Revenue"] * 100

    if "Net Income" in fin.columns:
        df["Net Margin"] = fin["Net Income"] / df["Revenue"] * 100

    return df.dropna(axis=1, how="all").sort_index()


def project_scenario(base_rev, years, growth, margin, start_year):
    rows = []
    rev = base_rev
    for i in range(1, years + 1):
        rev *= (1 + growth)
        rows.append({
            "Year": start_year + i,
            "Revenue": rev,
            "EBITDA_Margin": margin,
            "D_and_A": rev * 0.04,
            "CapEx": rev * 0.05,
            "Change_in_NWC": rev * 0.01
        })
    return pd.DataFrame(rows)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.title("GT Valuation Terminal")
    nav = st.radio(
        "Navigation",
        [
            "🗂️ Project Setup",
            "📈 Live Market Terminal",
            "💎 DCF & Scenario Analysis",
            "🌍 Comps Regression",
            "⚡ Risk & Reporting",
            "🏥 Financial Health (Z-Score)",
            "📊 Deep Dive"
        ]
    )

# --------------------------------------------------
# 5. PROJECT SETUP
# --------------------------------------------------
if nav == "🗂️ Project Setup":
    st.title("Project Setup")

    ticker = st.text_input("Ticker Symbol", "AAPL").upper()

    c1, c2, c3 = st.columns(3)
    with c1:
        bear_g = st.number_input("Bear Growth %", 2.0) / 100
        bear_m = st.number_input("Bear Margin %", 20.0) / 100
    with c2:
        base_g = st.number_input("Base Growth %", 5.0) / 100
        base_m = st.number_input("Base Margin %", 30.0) / 100
    with c3:
        bull_g = st.number_input("Bull Growth %", 10.0) / 100
        bull_m = st.number_input("Bull Margin %", 35.0) / 100

    if st.button("Build Scenarios"):
        mkt = get_market_data(ticker)
        base_rev = mkt["revenue"]
        year = datetime.now().year

        st.session_state["scenarios"] = {
            "Bear": project_scenario(base_rev, 5, bear_g, bear_m, year),
            "Base": project_scenario(base_rev, 5, base_g, base_m, year),
            "Bull": project_scenario(base_rev, 5, bull_g, bull_m, year)
        }
        st.session_state["ticker_data"] = mkt
        st.session_state["ticker_name"] = mkt["name"]
        st.session_state["ticker_symbol"] = ticker

        st.success("Scenarios Built Successfully")

# --------------------------------------------------
# 6. LIVE MARKET TERMINAL
# --------------------------------------------------
elif nav == "📈 Live Market Terminal":
    st.title("Live Market Terminal")

    ticker = st.text_input("Search Ticker")
    if ticker:
        data = get_market_data(ticker)
        st.session_state["ticker_data"] = data

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{data['currency']} {data['price']:.2f}")
        c2.metric("Market Cap", f"${data['mkt_cap']/1e9:.1f}B")
        c3.metric("Beta", f"{data['beta']:.2f}")
        c4.metric("Risk Free Rate", f"{data['rf_rate']:.2%}")

        hist = data["history"]
        fig = go.Figure(go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"]
        ))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 7. DCF & SCENARIOS
# --------------------------------------------------
elif nav == "💎 DCF & Scenario Analysis":
    st.title("DCF Valuation")

    wacc = st.slider("WACC %", 6.0, 14.0, 10.0) / 100
    tgr = st.slider("Terminal Growth %", 1.0, 4.0, 2.5) / 100
    st.session_state["wacc"] = wacc

    evs = {}
    for name, df in st.session_state["scenarios"].items():
        d = df.copy()
        d["EBITDA"] = d["Revenue"] * d["EBITDA_Margin"]
        d["NOPAT"] = (d["EBITDA"] - d["D_and_A"]) * (1 - 0.25)
        d["UFCF"] = d["NOPAT"] + d["D_and_A"] - d["CapEx"] - d["Change_in_NWC"]
        d["PV"] = d["UFCF"] / ((1 + wacc) ** np.arange(1, len(d) + 1))
        tv = (d["UFCF"].iloc[-1] * (1 + tgr)) / (wacc - tgr)
        evs[name] = d["PV"].sum() + tv / ((1 + wacc) ** len(d))
        if name == "Base":
            st.session_state["base_df"] = d

    st.session_state["ev_results"] = evs

    c1, c2, c3 = st.columns(3)
    c1.metric("Bear EV", f"${evs['Bear']:,.0f}")
    c2.metric("Base EV", f"${evs['Base']:,.0f}")
    c3.metric("Bull EV", f"${evs['Bull']:,.0f}")

# --------------------------------------------------
# 8. FINANCIAL HEALTH (Z-SCORE)
# --------------------------------------------------
elif nav == "🏥 Financial Health (Z-Score)":
    d = st.session_state["ticker_data"]
    sector = d.get("sector", "").lower()

    if any(x in sector for x in ["bank", "financial"]):
        st.error("Altman Z-Score not applicable for banks.")
    else:
        ta, tl = d["total_assets"], d["total_liab"]
        wc, re = d["working_capital"], d["retained_earnings"]
        ebit, rev = d["ebit"], d["revenue"]
        mkt_cap = d["mkt_cap"]

        z = (
            1.2 * (wc / ta)
            + 1.4 * (re / ta)
            + 3.3 * (ebit / ta)
            + 0.6 * (mkt_cap / tl)
            + 1.0 * (rev / ta)
        )

        st.metric("Altman Z-Score", f"{z:.2f}")

# --------------------------------------------------
# 9. DEEP DIVE
# --------------------------------------------------
elif nav == "📊 Deep Dive":
    st.title("Margin Trends")
    df = get_historical_profitability(st.session_state["ticker_symbol"])
    if not df.empty:
        fig = px.line(df, x=df.index, y=df.columns, markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.style.format("{:.2f}%"))
    else:
        st.warning("Margin data unavailable.")
