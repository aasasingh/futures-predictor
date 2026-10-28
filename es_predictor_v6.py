import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import warnings, os, sys, datetime, json
warnings.filterwarnings('ignore')

# Task Scheduler redirects stdout to a file with the cp1252 codec, which can't
# encode the box/check characters below
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

# ================================================================
#  ES PREDICTOR v6 — DUAL ENGINE
#
#  Engine 1: REGIME GATE
#    Detects market regime → only allows trades in regimes
#    where v5 proved accuracy > 55% (Range+Calm, Range+Normal,
#    Downtrend+Normal)
#
#  Engine 2: VOLATILITY PREDICTOR
#    Predicts whether next-week realized vol > current implied vol
#    Used for: position sizing + standalone vol trades
#
#  Combined signal:
#    Regime gate says WHEN to trade
#    Direction model says WHICH WAY
#    Vol model says HOW MUCH
#
#  Output: daily actionable signal for MES paper trading
# ================================================================
print("=" * 65)
print("  ES PREDICTOR v6 — DUAL ENGINE")
print("=" * 65)
_ensemble = ["GBM", "RF"]
if HAS_LGB: _ensemble.append("LGB")
if HAS_XGB: _ensemble.append("XGB")
print(f"  Engine 1:  Regime-gated direction  [{' + '.join(_ensemble)}]")
print(f"  Engine 2:  Volatility predictor    [{' + '.join(_ensemble)}]")
print(f"  Combined:  WHEN × DIRECTION × SIZE")
print(f"  Output:    Paper-trade ready MES signals")


# ================================================================
# PHASE 1: DATA PULL
# ================================================================
print(f"\n{'='*65}")
print("[1] Pulling market data...")
print(f"{'='*65}")

TICKERS = {
    "ES":    "ES=F",
    "VIX":   "^VIX",
    "VIX3M": "^VIX3M",
    "TNX":   "^TNX",
    "DXY":   "DX-Y.NYB",
    "NQ":    "NQ=F",
    "GC":    "GC=F",
    "CL":    "CL=F",
    "XLK":   "XLK",
    "XLF":   "XLF",
    "XLE":   "XLE",
    "XLV":   "XLV",
    "XLI":   "XLI",
    "XLU":   "XLU",
    "IWM":   "IWM",
    "HYG":   "HYG",
    "TLT":   "TLT",
}

# Today's bar is still trading (futures also print a bar on Sunday evening),
# so only completed sessions are used
today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

all_data = {}
for name, ticker in TICKERS.items():
    try:
        df = yf.Ticker(ticker).history(period="5y", interval="1d")
        # ^VIX and ^TNX come back in Chicago time, everything else in New York.
        # Reindexing tz-aware indexes across zones matched VIX to the previous
        # day, so strip to plain dates and align by trading day.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df[df.index < today]
        if len(df) > 100:
            all_data[name] = df
            print(f"  {name:>5} ({ticker:<12}): {len(df):>4} rows  ✓")
        else:
            print(f"  {name:>5} ({ticker:<12}): too short  ✗")
    except Exception as e:
        print(f"  {name:>5} ({ticker:<12}): FAILED — {e}")

data = all_data["ES"].copy()
es_dates = data.index


# ================================================================
# PHASE 2: FRED MACRO DATA
# ================================================================
print(f"\n{'='*65}")
print("[2] Pulling FRED macro data...")
print(f"{'='*65}")

FRED_SERIES = {
    "T10Y2Y":        "Yield curve 10Y-2Y",
    "T10Y3M":        "Yield curve 10Y-3M",
    "BAMLH0A0HYM2":  "HY OAS",
    "BAMLC0A4CBBB":  "BBB spread",
    "STLFSI2":       "Financial Stress",
    "ICSA":          "Jobless Claims",
    "EFFR":          "Fed Funds Rate",
    "DGS2":          "2Y yield",
    "DGS10":         "10Y yield",
}

fred_data = {}
FRED_KEY = os.environ.get("FRED_API_KEY", None)

if HAS_FRED and FRED_KEY:
    fred = Fred(api_key=FRED_KEY)
    for sid, desc in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start="2020-01-01")
            s.index = pd.to_datetime(s.index)
            s = s.reindex(es_dates, method="ffill")
            # FRED only serves 3 years of the ICE BofA series (HY OAS, BBB).
            # A short series would make dropna() throw away most of the
            # price history, so skip it (HYG covers credit anyway).
            if s.notna().mean() < 0.9:
                print(f"    {sid:<18} {desc:<25} {s.notna().sum():>4} valid  skipped, history too short")
                continue
            fred_data[sid] = s
            print(f"    {sid:<18} {desc:<25} {s.notna().sum():>4} valid  ✓")
        except Exception as e:
            print(f"    {sid:<18} {desc:<25} FAILED: {e}")
else:
    print("  ⚠ No FRED API key — run: $env:FRED_API_KEY=\"your_key\"")


# ================================================================
# PHASE 3: FEATURE ENGINEERING
# ================================================================
print(f"\n{'='*65}")
print("[3] Engineering features...")
print(f"{'='*65}")

# ── Returns ─────────────────────────────────────────────────
for p in [1, 2, 3, 5, 10, 20]:
    data[f"Ret_{p}d"] = data["Close"].pct_change(p)

# ── Moving averages ─────────────────────────────────────────
for w in [5, 10, 20, 50, 100, 200]:
    data[f"MA_{w}"] = data["Close"].rolling(w).mean()
    data[f"Px_vs_MA{w}"] = data["Close"] / data[f"MA_{w}"] - 1

data["MA_5_20"]   = data["MA_5"]  / data["MA_20"]
data["MA_10_50"]  = data["MA_10"] / data["MA_50"]
data["MA_20_100"] = data["MA_20"] / data["MA_100"]
data["MA_50_200"] = data["MA_50"] / data["MA_200"]

# ── Volatility (CRITICAL for vol model) ─────────────────────
for w in [5, 10, 20, 40, 60]:
    data[f"RealVol_{w}"] = data["Ret_1d"].rolling(w).std() * np.sqrt(252) * 100
    # Annualized realized vol in percentage terms

# Parkinson (high-low) vol — better estimator than close-close
data["ParkVol_20"] = np.sqrt(
    (1 / (4 * np.log(2))) * ((np.log(data["High"] / data["Low"])) ** 2).rolling(20).mean()
) * np.sqrt(252) * 100

# Vol of vol — how much is vol itself changing?
data["VolOfVol_20"] = data["RealVol_20"].rolling(20).std()

# Vol ratio (short vs long) — expansion/compression signal
data["VolRatio_5_20"]  = data["RealVol_5"]  / data["RealVol_20"]
data["VolRatio_5_60"]  = data["RealVol_5"]  / data["RealVol_60"]
data["VolRatio_20_60"] = data["RealVol_20"] / data["RealVol_60"]

# ── Volume ──────────────────────────────────────────────────
data["Vol_chg"]   = data["Volume"].pct_change()
data["Vol_MA20"]  = data["Volume"].rolling(20).mean()
data["Vol_spike"] = data["Volume"] / data["Vol_MA20"]

# ── RSI ─────────────────────────────────────────────────────
delta = data["Close"].diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
data["RSI"] = 100 - (100 / (1 + gain / loss))
data["RSI_MA5"]  = data["RSI"].rolling(5).mean()
data["RSI_div"]  = data["RSI"] - data["RSI_MA5"]

# ── MACD ────────────────────────────────────────────────────
e12 = data["Close"].ewm(span=12, adjust=False).mean()
e26 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"]      = e12 - e26
data["MACD_sig"]  = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_hist"] = data["MACD"] - data["MACD_sig"]

# ── Bollinger ───────────────────────────────────────────────
bb_mid = data["Close"].rolling(20).mean()
bb_std = data["Close"].rolling(20).std()
data["BB_pos"]   = (data["Close"] - (bb_mid - 2*bb_std)) / (4*bb_std)
data["BB_width"] = (4 * bb_std) / bb_mid

# ── ATR ─────────────────────────────────────────────────────
tr = pd.concat([
    data["High"] - data["Low"],
    abs(data["High"] - data["Close"].shift(1)),
    abs(data["Low"]  - data["Close"].shift(1)),
], axis=1).max(axis=1)
data["ATR_14"]    = tr.rolling(14).mean()
data["ATR_ratio"] = tr / data["ATR_14"]
data["ATR_pct"]   = data["ATR_14"] / data["Close"] * 100  # ATR as % of price

# ── Stochastic ──────────────────────────────────────────────
h14 = data["High"].rolling(14).max()
l14 = data["Low"].rolling(14).min()
data["Stoch_K"] = (data["Close"] - l14) / (h14 - l14 + 1e-9) * 100
data["Stoch_D"] = data["Stoch_K"].rolling(3).mean()

# ── ROC ─────────────────────────────────────────────────────
for p in [3, 7, 14, 21]:
    data[f"ROC_{p}"] = data["Close"].pct_change(p) * 100

# ── OBV ─────────────────────────────────────────────────────
obv_dir = np.sign(data["Close"].diff().fillna(0))
data["OBV"]       = (obv_dir * data["Volume"]).cumsum()
data["OBV_MA20"]  = data["OBV"].rolling(20).mean()
data["OBV_ratio"] = data["OBV"] / data["OBV_MA20"].replace(0, np.nan)

# ── Distance ────────────────────────────────────────────────
data["Dist_52w_hi"] = data["Close"] / data["Close"].rolling(252).max() - 1
data["Dist_52w_lo"] = data["Close"] / data["Close"].rolling(252).min() - 1

# ── Gap / range / candle ────────────────────────────────────
data["Gap_pct"]   = data["Open"] / data["Close"].shift(1) - 1
data["Range_pct"] = (data["High"] - data["Low"]) / data["Close"]
data["Body_size"] = abs(data["Close"] - data["Open"]) / data["Open"]
data["Is_green"]  = (data["Close"] > data["Open"]).astype(int)

# ── Streak ──────────────────────────────────────────────────
_c = data["Close"].values; _o = data["Open"].values
_str = np.zeros(len(_c), dtype=int); _cur = 0
for _i in range(len(_c)):
    _cur = (_cur + 1 if _cur > 0 else 1) if _c[_i] > _o[_i] else (_cur - 1 if _cur < 0 else -1)
    _str[_i] = _cur
data["Streak"] = _str

# ── Day of week ─────────────────────────────────────────────
data["DoW"] = data.index.dayofweek

# ── Lags ────────────────────────────────────────────────────
for lag in [1, 2, 3, 5]:
    data[f"Ret_lag{lag}"]  = data["Ret_1d"].shift(lag)
    data[f"RSI_lag{lag}"]  = data["RSI"].shift(lag)
    data[f"Vol_lag{lag}"]  = data["RealVol_5"].shift(lag)

print(f"  Price features: {len([c for c in data.columns if c not in all_data['ES'].columns])}")


# ================================================================
# PHASE 4: CROSS-MARKET + ALT DATA FEATURES
# ================================================================
print(f"\n{'='*65}")
print("[4] Cross-market + macro features...")
print(f"{'='*65}")

for name in ["VIX", "TNX", "DXY", "NQ", "GC", "CL"]:
    if name not in all_data:
        continue
    other = all_data[name]["Close"].reindex(es_dates, method="ffill")
    data[f"{name}_ret"]    = other.pct_change()
    data[f"{name}_mom5"]   = other.pct_change(5)
    data[f"{name}_corr20"] = data["Ret_1d"].rolling(20).corr(other.pct_change())
    data[f"{name}_z"]      = (other - other.rolling(252).mean()) / other.rolling(252).std()

# ── VIX features (critical for vol model) ───────────────────
if "VIX" in all_data:
    vix = all_data["VIX"]["Close"].reindex(es_dates, method="ffill")
    data["VIX_lvl"]     = vix
    data["VIX_vs_MA10"] = vix / vix.rolling(10).mean() - 1
    data["VIX_vs_MA20"] = vix / vix.rolling(20).mean() - 1
    data["VIX_hi"]      = (vix > 25).astype(int)
    data["VIX_ext"]     = (vix > 30).astype(int)
    data["VIX_pctrank"] = vix.rolling(252).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False
    )
    data["VIX_chg1d"]   = vix.pct_change(1)
    data["VIX_chg5d"]   = vix.pct_change(5)

    # VIX vs realized vol — the core vol trade signal
    # When VIX >> realized, vol is "expensive" (sell vol)
    # When VIX << realized, vol is "cheap" (buy vol)
    data["VRP_20"] = vix - data["RealVol_20"]   # Vol Risk Premium (20d)
    data["VRP_40"] = vix - data["RealVol_40"]   # Vol Risk Premium (40d)
    data["VRP_z"]  = (data["VRP_20"] - data["VRP_20"].rolling(252).mean()) / data["VRP_20"].rolling(252).std()

# ── VIX term structure ──────────────────────────────────────
if "VIX3M" in all_data and "VIX" in all_data:
    vix3m = all_data["VIX3M"]["Close"].reindex(es_dates, method="ffill")
    data["VIX_term"]       = vix / vix3m
    data["VIX_term_ma5"]   = data["VIX_term"].rolling(5).mean()
    data["VIX_contango"]   = (data["VIX_term"] < 0.95).astype(int)
    data["VIX_backwrd"]    = (data["VIX_term"] > 1.05).astype(int)
    print(f"  VIX term structure: ✓")

# ── TNX ─────────────────────────────────────────────────────
if "TNX" in all_data:
    tnx = all_data["TNX"]["Close"].reindex(es_dates, method="ffill")
    data["TNX_lvl"]   = tnx
    data["TNX_chg"]   = tnx.pct_change(1)
    data["TNX_mom5"]  = tnx.pct_change(5)

if "DXY" in all_data:
    dxy = all_data["DXY"]["Close"].reindex(es_dates, method="ffill")
    data["DXY_vs_MA20"] = dxy / dxy.rolling(20).mean() - 1

if "NQ" in all_data:
    nq = all_data["NQ"]["Close"].reindex(es_dates, method="ffill")
    data["ES_NQ_spr"]  = data["Ret_1d"] - nq.pct_change()
    data["NQ_rel_5d"]  = nq.pct_change(5) - data["Ret_5d"]

if "GC" in all_data:
    data["ES_GC_spr"] = data["Ret_1d"] - all_data["GC"]["Close"].reindex(es_dates, method="ffill").pct_change()

if "CL" in all_data:
    data["CL_mom5"] = all_data["CL"]["Close"].reindex(es_dates, method="ffill").pct_change(5)

# ── Breadth ─────────────────────────────────────────────────
sectors = [s for s in ["XLK","XLF","XLE","XLV","XLI","XLU"] if s in all_data]
if len(sectors) >= 4:
    sec_rets = pd.DataFrame({
        s: all_data[s]["Close"].reindex(es_dates, method="ffill").pct_change()
        for s in sectors
    })
    data["Sec_disp"]    = sec_rets.std(axis=1)
    data["Sec_disp_5d"] = data["Sec_disp"].rolling(5).mean()
    sec_mom = pd.DataFrame({
        s: all_data[s]["Close"].reindex(es_dates, method="ffill").pct_change(20)
        for s in sectors
    })
    data["Breadth_pct"] = (sec_mom > 0).sum(axis=1) / len(sectors)
    if "XLU" in all_data:
        xlu = all_data["XLU"]["Close"].reindex(es_dates, method="ffill")
        data["Def_rot"] = xlu.pct_change(20) - data["Ret_5d"].rolling(4).sum()
    print(f"  Breadth ({len(sectors)} sectors): ✓")

if "IWM" in all_data:
    iwm = all_data["IWM"]["Close"].reindex(es_dates, method="ffill")
    data["IWM_rel_5d"]  = iwm.pct_change(5) - data["Ret_5d"]
    data["IWM_rel_20d"] = iwm.pct_change(20) - data["Close"].pct_change(20)

if "HYG" in all_data:
    hyg = all_data["HYG"]["Close"].reindex(es_dates, method="ffill")
    data["HYG_mom5"]    = hyg.pct_change(5)
    data["HYG_mom20"]   = hyg.pct_change(20)
    data["HYG_corr_ES"] = data["Ret_1d"].rolling(20).corr(hyg.pct_change())
    if "TLT" in all_data:
        tlt_c = all_data["TLT"]["Close"].reindex(es_dates, method="ffill")
        data["HYG_vs_TLT"] = hyg.pct_change(5) - tlt_c.pct_change(5)

if "TLT" in all_data:
    tlt = all_data["TLT"]["Close"].reindex(es_dates, method="ffill")
    data["TLT_mom5"]       = tlt.pct_change(5)
    data["StockBond_corr"] = data["Ret_1d"].rolling(20).corr(tlt.pct_change())

# ── FRED macro ──────────────────────────────────────────────
if fred_data:
    print(f"  FRED macro features:")
    if "T10Y2Y" in fred_data:
        data["YC_10Y2Y"]      = fred_data["T10Y2Y"]
        data["YC_10Y2Y_chg"]  = fred_data["T10Y2Y"].diff(5)
        data["YC_inverted"]   = (fred_data["T10Y2Y"] < 0).astype(int)
        print(f"    Yield curve 10Y-2Y ✓")
    if "T10Y3M" in fred_data:
        data["YC_10Y3M"]      = fred_data["T10Y3M"]
        data["YC_10Y3M_chg"]  = fred_data["T10Y3M"].diff(5)
        print(f"    Yield curve 10Y-3M ✓")
    if "BAMLH0A0HYM2" in fred_data:
        hy = fred_data["BAMLH0A0HYM2"]
        data["HY_OAS"]        = hy
        data["HY_OAS_chg"]    = hy.diff(5)
        data["HY_OAS_z"]      = (hy - hy.rolling(252).mean()) / hy.rolling(252).std()
        data["HY_OAS_hi"]     = (hy > hy.rolling(252).quantile(0.75)).astype(int)
        print(f"    HY OAS ✓")
    if "BAMLC0A4CBBB" in fred_data:
        bbb = fred_data["BAMLC0A4CBBB"]
        data["BBB_spr"]       = bbb
        data["BBB_spr_chg"]   = bbb.diff(5)
        print(f"    BBB spread ✓")
    if "STLFSI2" in fred_data:
        st = fred_data["STLFSI2"]
        data["FinStress"]     = st
        data["FinStress_chg"] = st.diff(5)
        data["FinStress_hi"]  = (st > 0).astype(int)
        print(f"    Financial stress ✓")
    if "ICSA" in fred_data:
        ic = fred_data["ICSA"]
        data["Claims"]        = ic
        data["Claims_chg4w"]  = ic.pct_change(4)
        data["Claims_z"]      = (ic - ic.rolling(52).mean()) / ic.rolling(52).std()
        print(f"    Jobless claims ✓")
    if "EFFR" in fred_data:
        ef = fred_data["EFFR"]
        data["FedFunds"]      = ef
        data["FF_chg"]        = ef.diff(20)
        print(f"    Fed Funds ✓")
    if "DGS2" in fred_data and "DGS10" in fred_data:
        data["YC_slope"]      = fred_data["DGS10"] - fred_data["DGS2"]
        data["YC_slope_chg"]  = data["YC_slope"].diff(5)
        print(f"    Yield slope ✓")


# ================================================================
# PHASE 5: REGIME DETECTION
# ================================================================
print(f"\n{'='*65}")
print("[5] Regime detection...")
print(f"{'='*65}")

# Volatility regime
v20 = data["RealVol_20"] if "RealVol_20" in data.columns else data["Ret_1d"].rolling(20).std() * np.sqrt(252) * 100
vq25 = v20.rolling(60).quantile(0.25)
vq75 = v20.rolling(60).quantile(0.75)
data["Vol_regime"] = 1
data.loc[v20 < vq25, "Vol_regime"] = 0
data.loc[v20 > vq75, "Vol_regime"] = 2

# Trend regime
ma50_slope = data["MA_50"].pct_change(10)
data["Trend_regime"] = 1
data.loc[(data["Px_vs_MA50"] >  0.02) & (ma50_slope > 0), "Trend_regime"] = 2
data.loc[(data["Px_vs_MA50"] < -0.02) & (ma50_slope < 0), "Trend_regime"] = 0

# VIX regime
if "VIX_lvl" in data.columns:
    v = data["VIX_lvl"]
    data["VIX_regime"] = 1
    data.loc[v < 15, "VIX_regime"] = 0
    data.loc[v > 25, "VIX_regime"] = 2
    data.loc[v > 30, "VIX_regime"] = 3

# Credit regime
if "HY_OAS" in data.columns:
    ho = data["HY_OAS"]
    data["Credit_regime"] = 1
    data.loc[ho < ho.rolling(252).quantile(0.25), "Credit_regime"] = 0
    data.loc[ho > ho.rolling(252).quantile(0.75), "Credit_regime"] = 2

data["Regime_label"] = data["Trend_regime"].astype(str) + "_" + data["Vol_regime"].astype(str)

# FAVORABLE REGIMES (from v5 results: accuracy > 55%)
FAVORABLE_REGIMES = {"1_0", "1_1", "0_1"}
# 1_0 = Range + Calm (78.1%), 1_1 = Range + Normal (55.0%), 0_1 = Downtrend + Normal (55.6%)
data["Regime_favorable"] = data["Regime_label"].isin(FAVORABLE_REGIMES).astype(int)

# Interaction features
data["RSI_x_down"]    = data["RSI"] * (data["Trend_regime"] == 0).astype(int)
data["RSI_x_up"]      = data["RSI"] * (data["Trend_regime"] == 2).astype(int)
data["Mom5_x_vol"]    = data["Ret_5d"] * data["Vol_regime"]
data["BB_x_trend"]    = data["BB_pos"] * data["Trend_regime"]
data["Stoch_x_trend"] = data["Stoch_K"] * data["Trend_regime"]
if "VIX_regime" in data.columns:
    data["MACD_x_vix"] = data["MACD_hist"] * data["VIX_regime"]
if "Credit_regime" in data.columns:
    data["RSI_x_credit"] = data["RSI"] * data["Credit_regime"]
if "Breadth_pct" in data.columns:
    data["Breadth_x_trend"] = data["Breadth_pct"] * data["Trend_regime"]
if "VIX_term" in data.columns:
    data["VIXterm_x_trend"] = data["VIX_term"] * data["Trend_regime"]

trend_names = {0: "Downtrend", 1: "Range", 2: "Uptrend"}
vol_names   = {0: "Calm", 1: "Normal", 2: "Turbulent"}
cur_t = int(data["Trend_regime"].iloc[-1])
cur_v = int(data["Vol_regime"].iloc[-1])
print(f"  Current: {trend_names[cur_t]} + {vol_names[cur_v]}  (label: {data['Regime_label'].iloc[-1]})")
print(f"  Favorable: {'YES ✓' if data['Regime_favorable'].iloc[-1] else 'NO — sitting flat'}")
print(f"  Distinct regimes: {data['Regime_label'].nunique()}")


# ================================================================
# PHASE 6: TARGETS
# ================================================================
print(f"\n{'='*65}")
print("[6] Building targets...")
print(f"{'='*65}")

# ── Direction target (multi-horizon vote, same as v5) ────────
data["Tgt_1d"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data["Tgt_3d"] = (data["Close"].shift(-3) > data["Close"]).astype(int)
data["Tgt_5d"] = (data["Close"].shift(-5) > data["Close"]).astype(int)
data["Tgt_dir"] = ((data["Tgt_1d"] + data["Tgt_3d"] + data["Tgt_5d"]) >= 2).astype(int)

# ── Volatility target: will next 5-day realized vol exceed current VIX? ──
# This is the core vol trade: "is vol cheap or expensive right now?"
fwd_vol_5d = data["Ret_1d"].shift(-1).rolling(5).std().shift(-4) * np.sqrt(252) * 100
if "VIX_lvl" in data.columns:
    # 1 = realized vol will EXCEED implied (vol is cheap → buy vol)
    # 0 = realized vol will be BELOW implied (vol is expensive → sell vol)
    data["Tgt_vol"] = (fwd_vol_5d > data["VIX_lvl"]).astype(int)
else:
    # Fallback: will vol expand from current level?
    data["Tgt_vol"] = (fwd_vol_5d > data["RealVol_20"]).astype(int)

# ── Vol expansion target (simpler, more tradeable) ───────────
# Will next-week vol be higher or lower than this-week vol?
fwd_vol_week = data["Ret_1d"].shift(-1).rolling(5).std().shift(-4) * np.sqrt(252) * 100
cur_vol_week = data["Ret_1d"].rolling(5).std() * np.sqrt(252) * 100
data["Tgt_vol_expand"] = (fwd_vol_week > cur_vol_week * 1.1).astype(int)  # >10% vol expansion

# Comparisons against NaN come out False, so the last few rows were labelled 0
# instead of unknown. Blank them so they never get trained or scored on.
TARGETS = ["Tgt_1d", "Tgt_3d", "Tgt_5d", "Tgt_dir", "Tgt_vol", "Tgt_vol_expand"]
has_fwd_px  = data["Close"].shift(-5).notna()
has_fwd_vol = fwd_vol_5d.notna()
for c in ["Tgt_1d", "Tgt_3d", "Tgt_5d", "Tgt_dir"]:
    data[c] = data[c].where(has_fwd_px)
for c in ["Tgt_vol", "Tgt_vol_expand"]:
    data[c] = data[c].where(has_fwd_vol)

print(f"  Direction target up%:   {data['Tgt_dir'].mean():.1%}")
print(f"  Vol > VIX target up%:   {data['Tgt_vol'].mean():.1%}")
print(f"  Vol expand target up%:  {data['Tgt_vol_expand'].mean():.1%}")


# ================================================================
# PHASE 7: CLEAN
# ================================================================
data = data.replace([np.inf, -np.inf], np.nan)
# live_data keeps the newest rows (no outcome yet) for the live signal;
# data only keeps rows whose targets are known
live_data = data.dropna(subset=[c for c in data.columns if c not in TARGETS])
data = live_data.dropna(subset=TARGETS).copy()
data[TARGETS] = data[TARGETS].astype(int)
print(f"\n[7] Clean: {len(data)} rows with targets, live row: {live_data.index[-1].strftime('%Y-%m-%d')}")


# ================================================================
# PHASE 8: FEATURE SETUP
# ================================================================
print(f"\n{'='*65}")
print("[8] Feature setup...")
print(f"{'='*65}")

EXCLUDE = {
    "Open","High","Low","Close","Volume","Dividends","Stock Splits",
    "Tgt_dir","Tgt_1d","Tgt_3d","Tgt_5d","Tgt_vol","Tgt_vol_expand",
    "MA_5","MA_10","MA_20","MA_50","MA_100","MA_200",
    "Vol_MA20","VIX_lvl","TNX_lvl","ATR_14","Regime_label",
    "Regime_favorable","OBV","OBV_MA20",
}
feat_cols = [c for c in data.columns if c not in EXCLUDE]

print(f"  Feature candidates: {len(feat_cols)}")
X_all = data[feat_cols].values
y_dir = data["Tgt_dir"].values
y_vol = data["Tgt_vol"].values
dates = data.index


# ================================================================
# PHASE 9: WALK-FORWARD — DUAL ENGINE
# ================================================================
print(f"\n{'='*65}")
print(f"  ENGINE 1: REGIME-GATED DIRECTION MODEL")
print(f"  ENGINE 2: VOLATILITY PREDICTION MODEL")
print(f"{'='*65}")

TRAIN    = 250
TEST     = 40
PURGE    = 5
N_FEAT   = 35
HOLDOUT  = 126

# Split holdout
ho_X     = X_all[-HOLDOUT:]
ho_y_dir = y_dir[-HOLDOUT:]
ho_y_vol = y_vol[-HOLDOUT:]
ho_dates = dates[-HOLDOUT:]
ho_rets  = data["Ret_1d"].iloc[-HOLDOUT:].values
ho_reg   = data["Regime_label"].iloc[-HOLDOUT:].values
ho_fav   = data["Regime_favorable"].iloc[-HOLDOUT:].values

Xw      = X_all[:-HOLDOUT]
yw_dir  = y_dir[:-HOLDOUT]
yw_vol  = y_vol[:-HOLDOUT]
dw      = dates[:-HOLDOUT]
rw      = data["Ret_1d"].iloc[:-HOLDOUT].values
regw    = data["Regime_label"].iloc[:-HOLDOUT].values
favw    = data["Regime_favorable"].iloc[:-HOLDOUT].values

print(f"  Walk-forward: {len(Xw)} rows | Holdout: {HOLDOUT} rows")
print(f"  Holdout: {ho_dates[0].strftime('%Y-%m-%d')} → {ho_dates[-1].strftime('%Y-%m-%d')}")
print()

# Storage
dir_preds = []; dir_acts = []; dir_confs = []
vol_preds = []; vol_acts = []; vol_confs = []
all_dates_out = []; all_rets_out = []; all_regs_out = []; all_favs_out = []
regime_dir_preds = {}; regime_dir_acts = {}
dir_model_accs = {m: [] for m in _ensemble}
vol_model_accs = {m: [] for m in _ensemble}

last_dir_models = {}; last_vol_models = {}
last_scaler = None; last_feat_idx = None; last_feat_names = None
last_dir_meta = None; last_vol_meta = None

def train_stacking_ensemble(X_fit, y_fit, X_stack, y_stack, X_test, y_test, model_accs, label=""):
    """Train base models + stacking meta-learner. Returns (meta, models, probs, preds)."""
    models = {}
    stack_probs = []
    test_probs  = []

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42)
    gbm.fit(X_fit, y_fit)
    models["GBM"] = gbm
    stack_probs.append(gbm.predict_proba(X_stack)[:, 1])
    test_probs.append(gbm.predict_proba(X_test)[:, 1])
    model_accs["GBM"].append(accuracy_score(y_test, (test_probs[-1] > 0.5).astype(int)))

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        n_jobs=-1, random_state=42)
    rf.fit(X_fit, y_fit)
    models["RF"] = rf
    stack_probs.append(rf.predict_proba(X_stack)[:, 1])
    test_probs.append(rf.predict_proba(X_test)[:, 1])
    model_accs["RF"].append(accuracy_score(y_test, (test_probs[-1] > 0.5).astype(int)))

    if HAS_LGB:
        lgbm = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_child_samples=20, verbose=-1, random_state=42)
        lgbm.fit(X_fit, y_fit)
        models["LGB"] = lgbm
        stack_probs.append(lgbm.predict_proba(X_stack)[:, 1])
        test_probs.append(lgbm.predict_proba(X_test)[:, 1])
        model_accs["LGB"].append(accuracy_score(y_test, (test_probs[-1] > 0.5).astype(int)))

    if HAS_XGB:
        xgbm = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_child_weight=20, verbosity=0,
            eval_metric='logloss', random_state=42)
        xgbm.fit(X_fit, y_fit)
        models["XGB"] = xgbm
        stack_probs.append(xgbm.predict_proba(X_stack)[:, 1])
        test_probs.append(xgbm.predict_proba(X_test)[:, 1])
        model_accs["XGB"].append(accuracy_score(y_test, (test_probs[-1] > 0.5).astype(int)))

    # Stack
    sX_train = np.column_stack(stack_probs)
    sX_test  = np.column_stack(test_probs)
    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(sX_train, y_stack)
    ens_probs = meta.predict_proba(sX_test)[:, 1]
    ens_preds = (ens_probs > 0.5).astype(int)

    return meta, models, ens_probs, ens_preds


step = 0; i = 0
while i + TRAIN + PURGE + TEST <= len(Xw):
    te = i + TRAIN
    ts = te + PURGE
    tend = ts + TEST

    Xtr_raw = Xw[i:te];     Xte_raw = Xw[ts:tend]
    ytr_dir = yw_dir[i:te];  yte_dir = yw_dir[ts:tend]
    ytr_vol = yw_vol[i:te];  yte_vol = yw_vol[ts:tend]
    t_dates = dw[ts:tend];   t_rets  = rw[ts:tend]
    t_regs  = regw[ts:tend]; t_favs  = favw[ts:tend]

    # Per-fold feature selection (on direction target — shared features)
    mi = mutual_info_classif(Xtr_raw, ytr_dir, random_state=42)
    ranked = sorted(zip(feat_cols, mi, range(len(feat_cols))), key=lambda x: x[1], reverse=True)
    f_names = [r[0] for r in ranked[:N_FEAT]]
    f_idx   = [r[2] for r in ranked[:N_FEAT]]

    Xtr = Xtr_raw[:, f_idx]; Xte = Xte_raw[:, f_idx]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # 80/20 split for stacking
    sp = int(len(Xtr_s) * 0.8)
    Xf, Xs = Xtr_s[:sp], Xtr_s[sp:]
    yf_dir, ys_dir = ytr_dir[:sp], ytr_dir[sp:]
    yf_vol, ys_vol = ytr_vol[:sp], ytr_vol[sp:]

    # Train both engines
    dir_meta, dir_models, dir_p, dir_pd = train_stacking_ensemble(
        Xf, yf_dir, Xs, ys_dir, Xte_s, yte_dir, dir_model_accs, "DIR")
    vol_meta, vol_models, vol_p, vol_pd = train_stacking_ensemble(
        Xf, yf_vol, Xs, ys_vol, Xte_s, yte_vol, vol_model_accs, "VOL")

    # Accuracy — direction only in favorable regimes
    fav_mask = t_favs == 1
    dir_acc_all = accuracy_score(yte_dir, dir_pd)
    dir_acc_fav = accuracy_score(yte_dir[fav_mask], dir_pd[fav_mask]) if sum(fav_mask) >= 3 else np.nan
    vol_acc     = accuracy_score(yte_vol, vol_pd)

    # Save last
    last_dir_models = dir_models; last_vol_models = vol_models
    last_scaler = scaler; last_feat_idx = f_idx; last_feat_names = f_names
    last_dir_meta = dir_meta; last_vol_meta = vol_meta

    step += 1
    n_fav = sum(fav_mask)
    dom = pd.Series(t_regs).mode().iloc[0]
    ps = t_dates[0].strftime("%Y-%m-%d")
    pe = t_dates[-1].strftime("%Y-%m-%d")

    dir_fav_str = f"{dir_acc_fav:.0%}" if not np.isnan(dir_acc_fav) else "n/a"
    print(f"  [{step:>2}] {ps}→{pe}  Dir:{dir_acc_all:.0%}(all) {dir_fav_str}(fav:{n_fav}d)  Vol:{vol_acc:.0%}  {dom}")

    dir_preds.extend(dir_pd); dir_acts.extend(yte_dir); dir_confs.extend(dir_p)
    vol_preds.extend(vol_pd); vol_acts.extend(yte_vol); vol_confs.extend(vol_p)
    all_dates_out.extend(t_dates); all_rets_out.extend(t_rets)
    all_regs_out.extend(t_regs); all_favs_out.extend(t_favs)

    for idx in range(len(yte_dir)):
        r = t_regs[idx]
        regime_dir_preds.setdefault(r, []).append(dir_pd[idx])
        regime_dir_acts.setdefault(r, []).append(yte_dir[idx])

    i += TEST

dir_preds = np.array(dir_preds); dir_acts = np.array(dir_acts); dir_confs = np.array(dir_confs)
vol_preds = np.array(vol_preds); vol_acts = np.array(vol_acts); vol_confs = np.array(vol_confs)
all_rets  = np.array(all_rets_out); all_regs = np.array(all_regs_out); all_favs = np.array(all_favs_out)

# Overall stats
dir_acc_total = accuracy_score(dir_acts, dir_preds)
fav_mask_all  = all_favs == 1
dir_acc_fav_total = accuracy_score(dir_acts[fav_mask_all], dir_preds[fav_mask_all]) if sum(fav_mask_all) > 10 else 0
vol_acc_total = accuracy_score(vol_acts, vol_preds)

print(f"\n  ═══════════════════════════════════════════")
print(f"  DIRECTION MODEL:")
print(f"    All regimes:       {dir_acc_total:.2%}  ({len(dir_preds)} predictions)")
print(f"    Favorable only:    {dir_acc_fav_total:.2%}  ({sum(fav_mask_all)} predictions)")
print(f"  VOLATILITY MODEL:    {vol_acc_total:.2%}  ({len(vol_preds)} predictions)")
print(f"  ═══════════════════════════════════════════")

print(f"\n  Direction model per-component:")
for m, accs in dir_model_accs.items():
    print(f"    {m:>4}: {np.mean(accs):.2%}")
print(f"\n  Volatility model per-component:")
for m, accs in vol_model_accs.items():
    print(f"    {m:>4}: {np.mean(accs):.2%}")


# ================================================================
# PHASE 10: REGIME-SPECIFIC DIRECTION ACCURACY
# ================================================================
print(f"\n{'='*65}")
print("  DIRECTION ACCURACY BY REGIME")
print(f"{'='*65}")

RNAMES = {
    "0_0":"Down+Calm","0_1":"Down+Normal","0_2":"Down+Turbulent",
    "1_0":"Range+Calm","1_1":"Range+Normal","1_2":"Range+Turbulent",
    "2_0":"Up+Calm","2_1":"Up+Normal","2_2":"Up+Turbulent",
}

for r in sorted(regime_dir_preds.keys()):
    p = np.array(regime_dir_preds[r]); a = np.array(regime_dir_acts[r])
    if len(a) < 5: continue
    ac = accuracy_score(a, p)
    fav = "★" if r in FAVORABLE_REGIMES else " "
    flag = "✓" if ac > 0.55 else ("✗" if ac < 0.48 else "~")
    print(f"  {fav}{flag} {RNAMES.get(r,r):<25} {ac:.1%}  ({len(a):>3} days)")


# ================================================================
# PHASE 11: COMBINED BACKTEST — REGIME-GATED + VOL-SIZED
# ================================================================
print(f"\n{'='*65}")
print("  COMBINED BACKTEST — REGIME GATE + VOL SIZING")
print(f"{'='*65}")

TXN = 0.0001
capital = 100_000
eq = [float(capital)]
trades = []

for t in range(len(dir_preds) - 1):
    d_conf = dir_confs[t]
    v_conf = vol_confs[t]
    ret_nxt = all_rets[t + 1]
    regime = all_regs[t]
    fav = all_favs[t]

    # GATE 1: Only trade in favorable regimes
    if not fav:
        eq.append(eq[-1])
        continue

    # GATE 2: Need minimum directional confidence
    if 0.45 < d_conf < 0.55:
        eq.append(eq[-1])
        continue

    # Direction
    direction = 1.0 if d_conf > 0.55 else -1.0

    # Size based on vol prediction
    # If vol model says vol is expanding → reduce size (uncertain environment)
    # If vol model says vol is compressing → increase size (stable environment)
    if v_conf > 0.6:  # high confidence vol will expand
        vol_mult = 0.5   # reduce exposure — choppy ahead
    elif v_conf < 0.4:  # high confidence vol will compress
        vol_mult = 1.5   # increase exposure — calm ahead
    else:
        vol_mult = 1.0

    # Confidence sizing
    conf_str = abs(d_conf - 0.5)
    conf_mult = 1.0 + min(conf_str, 0.20) * 5.0

    size = float(np.clip(conf_mult * vol_mult, 0.3, 2.5))

    # Drawdown guard
    peak = max(eq)
    if (eq[-1] - peak) / peak < -0.08:
        size *= 0.5

    # Execute
    gross = ret_nxt * size * direction
    net   = gross - 2 * TXN * size
    if net < -0.015 * size:
        net = -0.015 * size

    eq.append(eq[-1] * (1 + net))
    trades.append({"ret": net, "dir": direction, "d_conf": d_conf,
                    "v_conf": v_conf, "size": size, "regime": regime})

eq = np.array(eq)
tot_ret = eq[-1] / eq[0] - 1
pk = np.maximum.accumulate(eq)
dd = ((eq - pk) / pk).min()

bh_eq = capital * np.cumprod(np.r_[1, 1 + all_rets[1:]])
bh_ret = bh_eq[-1] / bh_eq[0] - 1
bh_dd  = ((bh_eq - np.maximum.accumulate(bh_eq)) / np.maximum.accumulate(bh_eq)).min()

tr = np.array([t["ret"] for t in trades])
longs  = [t for t in trades if t["dir"] == 1]
shorts = [t for t in trades if t["dir"] == -1]
wins = tr[tr > 0]; losses = tr[tr < 0]
wr = len(wins) / len(tr) * 100 if len(tr) > 0 else 0
aw = wins.mean() if len(wins) > 0 else 0
al = abs(losses.mean()) if len(losses) > 0 else 0
pf = aw / al if al > 0 else 0

dr = np.diff(eq) / eq[:-1]
sh = np.mean(dr) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0
sd_down = np.std(dr[dr < 0]) if any(dr < 0) else 1e-9
so = np.mean(dr) / sd_down * np.sqrt(252)
ny = len(eq) / 252
ar = (eq[-1] / eq[0]) ** (1/ny) - 1 if ny > 0 else 0
ca = ar / abs(dd) if dd != 0 else 0

# Selectivity
total_days = len(dir_preds) - 1
trade_days = len(tr)
skip_pct   = (1 - trade_days / total_days) * 100

print(f"""
  Starting capital:  ${capital:>12,.0f}
  Final equity:      ${eq[-1]:>12,.0f}
  Total return:      {tot_ret:>10.2%}
  Buy & hold:        {bh_ret:>10.2%}

  Trade days:        {trade_days:>6} / {total_days}  ({skip_pct:.0f}% skipped — regime gated)
  Long / Short:      {len(longs)} / {len(shorts)}
  Win rate:          {wr:>8.1f}%
  Avg win:           {aw:>10.5f}
  Avg loss:          {al:>10.5f}
  Profit factor:     {pf:>10.2f}

  Max drawdown:      {dd:>10.2%}   (B&H: {bh_dd:.2%})
  Sharpe ratio:      {sh:>10.2f}
  Sortino ratio:     {so:>10.2f}
  Calmar ratio:      {ca:>10.2f}
  Ann. return:       {ar:>10.2%}

  {'▲ MODEL BEATS B&H' if tot_ret > bh_ret else '▼ B&H STILL WINS'}   Δ = {tot_ret - bh_ret:+.2%}
""")


# ================================================================
# PHASE 12: HOLDOUT VALIDATION
# ================================================================
print(f"\n{'='*65}")
print("  HOLDOUT — 6 MONTHS UNSEEN")
print(f"{'='*65}")

if last_dir_models and last_scaler is not None:
    ho_Xf = ho_X[:, last_feat_idx]
    ho_Xs = last_scaler.transform(ho_Xf)

    # Direction
    h_dir_bp = [last_dir_models[m].predict_proba(ho_Xs)[:, 1] for m in _ensemble if m in last_dir_models]
    h_dir_p  = last_dir_meta.predict_proba(np.column_stack(h_dir_bp))[:, 1]
    h_dir_pd = (h_dir_p > 0.5).astype(int)

    # Vol
    h_vol_bp = [last_vol_models[m].predict_proba(ho_Xs)[:, 1] for m in _ensemble if m in last_vol_models]
    h_vol_p  = last_vol_meta.predict_proba(np.column_stack(h_vol_bp))[:, 1]
    h_vol_pd = (h_vol_p > 0.5).astype(int)

    h_dir_acc = accuracy_score(ho_y_dir, h_dir_pd)
    h_vol_acc = accuracy_score(ho_y_vol, h_vol_pd)

    h_fav = ho_fav == 1
    h_dir_fav_acc = accuracy_score(ho_y_dir[h_fav], h_dir_pd[h_fav]) if sum(h_fav) >= 5 else 0

    # Holdout combined backtest
    h_eq = [100_000.0]
    h_trades = 0
    for t in range(len(h_dir_pd) - 1):
        if not ho_fav[t]:
            h_eq.append(h_eq[-1]); continue
        if 0.45 < h_dir_p[t] < 0.55:
            h_eq.append(h_eq[-1]); continue
        d = 1.0 if h_dir_p[t] > 0.55 else -1.0
        vm = 0.5 if h_vol_p[t] > 0.6 else (1.5 if h_vol_p[t] < 0.4 else 1.0)
        sz = float(np.clip(vm, 0.3, 2.5))
        net = ho_rets[t+1] * sz * d - 2 * TXN * sz
        if net < -0.015 * sz: net = -0.015 * sz
        h_eq.append(h_eq[-1] * (1 + net))
        h_trades += 1

    h_eq = np.array(h_eq)
    h_ret = h_eq[-1] / h_eq[0] - 1
    h_bh  = capital * np.cumprod(np.r_[1, 1 + ho_rets[1:]])
    h_bh_ret = h_bh[-1] / h_bh[0] - 1

    print(f"  Period:          {ho_dates[0].strftime('%Y-%m-%d')} → {ho_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Dir accuracy:    {h_dir_acc:.2%} (all) | {h_dir_fav_acc:.2%} (favorable: {sum(h_fav)} days)")
    print(f"  Vol accuracy:    {h_vol_acc:.2%}")
    print(f"  Combined return: {h_ret:.2%}  (B&H: {h_bh_ret:.2%})  Trades: {h_trades}")
    print(f"  {'▲ BEATS B&H' if h_ret > h_bh_ret else '▼ B&H WINS'}")


# ================================================================
# PHASE 13: STATISTICAL TESTS
# ================================================================
print(f"\n{'='*65}")
print("  STATISTICAL SIGNIFICANCE")
print(f"{'='*65}")

# Tested against the majority-class rate, not 50%. ES goes up more often than
# not, so "always long" alone would clear a 50% bar.
def majority(y):
    return max(np.mean(y), 1 - np.mean(y)) if len(y) else 0.5

# Direction — all
n_d = len(dir_preds); nc_d = sum(dir_preds == dir_acts)
base_d = majority(dir_acts)
p_d = stats.binomtest(nc_d, n_d, base_d, alternative="greater").pvalue

# Direction — favorable only
n_df = sum(fav_mask_all); nc_df = sum(dir_preds[fav_mask_all] == dir_acts[fav_mask_all])
base_df = majority(dir_acts[fav_mask_all])
p_df = stats.binomtest(nc_df, n_df, base_df, alternative="greater").pvalue if n_df > 10 else 1.0

# Vol
n_v = len(vol_preds); nc_v = sum(vol_preds == vol_acts)
base_v = majority(vol_acts)
p_v = stats.binomtest(nc_v, n_v, base_v, alternative="greater").pvalue

print(f"  Direction (all):       {dir_acc_total:.2%}  vs base {base_d:.2%}  p = {p_d:.4f}  {'✓ SIG' if p_d < 0.05 else '✗ NOT SIG'}")
print(f"  Direction (favorable): {dir_acc_fav_total:.2%}  vs base {base_df:.2%}  p = {p_df:.4f}  {'✓ SIG' if p_df < 0.05 else '✗ NOT SIG'}")
print(f"  Volatility:            {vol_acc_total:.2%}  vs base {base_v:.2%}  p = {p_v:.4f}  {'✓ SIG' if p_v < 0.05 else '✗ NOT SIG'}")


# ================================================================
# PHASE 14: LIVE SIGNAL
# ================================================================
print(f"\n{'='*65}")
print("  LIVE SIGNAL — NEXT TRADING DAY")
print(f"{'='*65}")

if last_dir_models and last_scaler is not None:
    lr = live_data[last_feat_names].iloc[-1:].values
    lr_s = last_scaler.transform(lr)

    # Direction probs
    live_dir_bp = {m: mdl.predict_proba(lr_s)[0][1]
                   for m, mdl in last_dir_models.items()}
    live_dir_stack = np.array([[v for v in live_dir_bp.values()]])
    live_dir_p = last_dir_meta.predict_proba(live_dir_stack)[0][1]

    # Vol probs
    live_vol_bp = {m: mdl.predict_proba(lr_s)[0][1]
                   for m, mdl in last_vol_models.items()}
    live_vol_stack = np.array([[v for v in live_vol_bp.values()]])
    live_vol_p = last_vol_meta.predict_proba(live_vol_stack)[0][1]

    # Current regime
    cur_regime = live_data["Regime_label"].iloc[-1]
    cur_fav    = cur_regime in FAVORABLE_REGIMES
    cur_t_name = trend_names[int(live_data["Trend_regime"].iloc[-1])]
    cur_v_name = vol_names[int(live_data["Vol_regime"].iloc[-1])]

    # Generate signal
    if not cur_fav:
        signal  = "⏸  NO TRADE"
        reason  = f"Regime ({cur_t_name}+{cur_v_name}) not in favorable set"
        action  = "FLAT"
        size_rec = 0
    elif live_dir_p > 0.55:
        signal  = "📈 LONG MES"
        reason  = f"Favorable regime + bullish signal ({live_dir_p:.1%})"
        action  = "BUY"
        vm = 0.5 if live_vol_p > 0.6 else (1.5 if live_vol_p < 0.4 else 1.0)
        size_rec = int(np.clip(vm * 2, 1, 4))  # number of MES contracts
    elif live_dir_p < 0.45:
        signal  = "📉 SHORT MES"
        reason  = f"Favorable regime + bearish signal ({live_dir_p:.1%})"
        action  = "SELL"
        vm = 0.5 if live_vol_p > 0.6 else (1.5 if live_vol_p < 0.4 else 1.0)
        size_rec = int(np.clip(vm * 2, 1, 4))
    else:
        signal  = "⏸  NO TRADE"
        reason  = f"Favorable regime but low directional confidence ({live_dir_p:.1%})"
        action  = "FLAT"
        size_rec = 0

    vol_outlook = "EXPANDING ⚠" if live_vol_p > 0.6 else ("COMPRESSING ✓" if live_vol_p < 0.4 else "NEUTRAL")

    print(f"""
  ┌─────────────────────────────────────────────┐
  │  SIGNAL:    {signal:<33}│
  │  ACTION:    {action:<33}│
  │  SIZE:      {size_rec} MES contracts{' '*(20-len(str(size_rec)))}│
  │  REASON:    {reason[:33]:<33}│
  └─────────────────────────────────────────────┘

  Regime:       {cur_t_name} + {cur_v_name}  {'(FAVORABLE ★)' if cur_fav else '(UNFAVORABLE — sitting out)'}
  Dir model:    {live_dir_p:.1%} (>{'>'}0.55=long, <0.45=short)
  Vol outlook:  {vol_outlook}  ({live_vol_p:.1%} prob of vol expansion)
  Data through: {live_data.index[-1].strftime('%Y-%m-%d')}

  Base models (direction):  {' | '.join(f'{k}:{v:.1%}' for k,v in live_dir_bp.items())}
  Base models (vol):        {' | '.join(f'{k}:{v:.1%}' for k,v in live_vol_bp.items())}
""")


# ================================================================
# PHASE 15: PAPER TRADE LOGGER
# ================================================================
log_file = "es_v6_signals.csv"
log = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "data_date": live_data.index[-1].strftime("%Y-%m-%d"),
    "signal": action if last_dir_models else "N/A",
    "size_mes": size_rec if last_dir_models else 0,
    "dir_prob": f"{live_dir_p:.4f}" if last_dir_models else "",
    "vol_prob": f"{live_vol_p:.4f}" if last_dir_models else "",
    "regime": cur_regime if last_dir_models else "",
    "regime_favorable": cur_fav if last_dir_models else "",
    "vol_outlook": vol_outlook if last_dir_models else "",
}
for m in _ensemble:
    log[f"dir_{m}"] = f"{live_dir_bp.get(m,0):.4f}" if last_dir_models else ""
    log[f"vol_{m}"] = f"{live_vol_bp.get(m,0):.4f}" if last_dir_models else ""

try:
    old = pd.read_csv(log_file)
    new = pd.concat([old, pd.DataFrame([log])], ignore_index=True)
except FileNotFoundError:
    new = pd.DataFrame([log])
new.to_csv(log_file, index=False)
print(f"  📋 Signal logged to {log_file}")
print(f"     Run this script daily. After 30+ entries, compare signal vs actual.")


# ================================================================
# FINAL VERDICT
# ================================================================
print(f"\n{'='*65}")
print("  FINAL VERDICT — v6 DUAL ENGINE")
print(f"{'='*65}")
print(f"""
  ENGINE 1 — DIRECTION (regime-gated):
    All regimes:     {dir_acc_total:.2%}   (p = {p_d:.4f})
    Favorable only:  {dir_acc_fav_total:.2%}   (p = {p_df:.4f})
    Holdout (fav):   {h_dir_fav_acc:.2%}

  ENGINE 2 — VOLATILITY:
    Walk-forward:    {vol_acc_total:.2%}   (p = {p_v:.4f})
    Holdout:         {h_vol_acc:.2%}

  COMBINED BACKTEST:
    Return:          {tot_ret:.2%}  (B&H: {bh_ret:.2%})
    Sharpe:          {sh:.2f}
    Max drawdown:    {dd:.2%}
    Profit factor:   {pf:.2f}
    Selectivity:     {skip_pct:.0f}% of days skipped

  HOLDOUT:
    Return:          {h_ret:.2%}  (B&H: {h_bh_ret:.2%})

  PAPER TRADING RULES:
    1. Run this script daily before market open
    2. Only trade when signal says LONG or SHORT
    3. Use recommended MES contract count
    4. Log actual results in a separate spreadsheet
    5. After 30 trades, compare win rate vs model prediction
    6. If win rate > 55% after 30 trades → go live with real capital
    7. If win rate < 50% → model needs more work

  WHAT EACH SIGNAL MEANS:
    LONG MES  = Buy micro E-mini S&P 500 futures at open
    SHORT MES = Sell micro E-mini S&P 500 futures at open
    FLAT      = No position — regime unfavorable or low confidence
    SIZE      = Number of MES contracts (1 MES = $5 × ES price ≈ $28k notional)
""")
