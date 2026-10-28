import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Optional boosting libraries — graceful fallback if missing
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

# ============================================================
#  ES FUTURES PREDICTOR v4 — ENSEMBLE EDITION
#
#  v4 improvements over v3:
#   ✓ Fix: MI feature selection moved inside warmup window (no leak)
#   ✓ Multi-horizon target: majority vote 1d + 3d + 5d (less noise)
#   ✓ New features: ROC, Stochastic, OBV, 52w levels, gap, regime interactions
#   ✓ Ensemble: GBM + RF + LightGBM + XGBoost (avg probs)
#   ✓ Kelly criterion position sizing (fractional 25%)
#   ✓ Transaction costs: 0.01% per side
#   ✓ Drawdown guard: cut size 50% when DD > 10%
#   ✓ Calmar + Sortino ratios
#   ✓ Live next-day signal at the end
# ============================================================
print("=" * 62)
print("  ES FUTURES PREDICTOR v4 — ENSEMBLE EDITION")
print("=" * 62)
_ensemble_names = ["GBM", "RF"]
if HAS_LGB: _ensemble_names.append("LGB")
if HAS_XGB: _ensemble_names.append("XGB")
print(f"  Ensemble:  {' + '.join(_ensemble_names)}")
print(f"  Target:    multi-horizon majority vote (1d + 3d + 5d)")
print(f"  Risk:      Kelly sizing | txn costs | drawdown guard")


# ============================================================
# PHASE 1: PULL MULTI-MARKET DATA
# ============================================================
print("\n[1] Pulling market data...")

TICKERS = {
    "ES":  "ES=F",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
    "NQ":  "NQ=F",
    "GC":  "GC=F",
    "CL":  "CL=F",
}

all_data = {}
for name, ticker in TICKERS.items():
    try:
        df = yf.Ticker(ticker).history(period="5y", interval="1d")
        all_data[name] = df
        print(f"  {name:>4} ({ticker:<10}): {len(df):>4} rows")
    except Exception as e:
        print(f"  {name:>4} FAILED: {e}")

data = all_data["ES"].copy()


# ============================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================
print("\n[2] Engineering features...")

# ── Returns ──────────────────────────────────────────────────
for period in [1, 2, 3, 5, 10]:
    data[f"Return_{period}d"] = data["Close"].pct_change(period)

# ── Moving averages ───────────────────────────────────────────
for window in [5, 10, 20, 50, 100]:
    data[f"MA_{window}"]         = data["Close"].rolling(window).mean()
    data[f"Price_vs_MA{window}"] = data["Close"] / data[f"MA_{window}"] - 1

data["MA_5_20_cross"]   = data["MA_5"]  / data["MA_20"]
data["MA_10_50_cross"]  = data["MA_10"] / data["MA_50"]
data["MA_20_100_cross"] = data["MA_20"] / data["MA_100"]

# ── Volatility ────────────────────────────────────────────────
data["Volatility_5"]  = data["Return_1d"].rolling(5).std()
data["Volatility_10"] = data["Return_1d"].rolling(10).std()
data["Volatility_20"] = data["Return_1d"].rolling(20).std()
data["Vol_expansion"] = data["Volatility_5"] / data["Volatility_20"]

# ── Volume ────────────────────────────────────────────────────
data["Vol_change"] = data["Volume"].pct_change()
data["Vol_MA5"]    = data["Volume"].rolling(5).mean()
data["Vol_MA20"]   = data["Volume"].rolling(20).mean()
data["Vol_spike"]  = data["Volume"] / data["Vol_MA20"]

# ── RSI ───────────────────────────────────────────────────────
delta = data["Close"].diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
data["RSI"]     = 100 - (100 / (1 + gain / loss))
data["RSI_MA"]  = data["RSI"].rolling(5).mean()
data["RSI_div"] = data["RSI"] - data["RSI_MA"]   # momentum within RSI

# ── MACD ──────────────────────────────────────────────────────
ema12 = data["Close"].ewm(span=12, adjust=False).mean()
ema26 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"]            = ema12 - ema26
data["MACD_signal"]     = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_hist"]       = data["MACD"] - data["MACD_signal"]
data["MACD_hist_chg"]   = data["MACD_hist"].diff()   # histogram slope

# ── Bollinger Bands ───────────────────────────────────────────
bb_mid  = data["Close"].rolling(20).mean()
bb_std  = data["Close"].rolling(20).std()
data["BB_position"] = (data["Close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std)
data["BB_width"]    = (4 * bb_std) / bb_mid           # bandwidth = vol proxy

# ── ATR ───────────────────────────────────────────────────────
true_range = pd.concat([
    data["High"] - data["Low"],
    abs(data["High"] - data["Close"].shift(1)),
    abs(data["Low"]  - data["Close"].shift(1)),
], axis=1).max(axis=1)
data["ATR_14"]    = true_range.rolling(14).mean()
data["ATR_ratio"] = true_range / data["ATR_14"]

# ── Stochastic Oscillator ─────────────────────────────────────
high_14 = data["High"].rolling(14).max()
low_14  = data["Low"].rolling(14).min()
data["Stoch_K"] = (data["Close"] - low_14) / (high_14 - low_14 + 1e-9) * 100
data["Stoch_D"] = data["Stoch_K"].rolling(3).mean()

# ── Rate of Change (multi-speed momentum) ────────────────────
for period in [3, 7, 14, 21]:
    data[f"ROC_{period}"] = data["Close"].pct_change(period) * 100

# ── OBV (On-Balance Volume) ───────────────────────────────────
obv_dir       = np.sign(data["Close"].diff().fillna(0))
data["OBV"]   = (obv_dir * data["Volume"]).cumsum()
data["OBV_MA20"]  = data["OBV"].rolling(20).mean()
# Ratio relative to its own MA — OBV level is meaningless in isolation
data["OBV_ratio"] = data["OBV"] / data["OBV_MA20"].replace(0, np.nan)

# ── 52-week distance (mean-reversion / momentum context) ──────
data["Dist_52w_high"] = data["Close"] / data["Close"].rolling(252).max() - 1
data["Dist_52w_low"]  = data["Close"] / data["Close"].rolling(252).min() - 1

# ── Overnight gap ─────────────────────────────────────────────
data["Gap_pct"]    = data["Open"] / data["Close"].shift(1) - 1

# ── Intraday range ────────────────────────────────────────────
data["Range_pct"]  = (data["High"] - data["Low"]) / data["Close"]

# ── Candle structure ──────────────────────────────────────────
data["Body_size"]   = abs(data["Close"] - data["Open"]) / data["Open"]
data["Upper_wick"]  = (data["High"] - data[["Close", "Open"]].max(axis=1)) / data["Open"]
data["Lower_wick"]  = (data[["Close", "Open"]].min(axis=1) - data["Low"]) / data["Open"]
data["Is_green"]    = (data["Close"] > data["Open"]).astype(int)

# ── Win/loss streak (sequential) ─────────────────────────────
# Inherently sequential — compute with a tight loop over numpy arrays
_close = data["Close"].values
_open  = data["Open"].values
_streaks = np.zeros(len(_close), dtype=int)
_cur = 0
for _i in range(len(_close)):
    if _close[_i] > _open[_i]:
        _cur = _cur + 1 if _cur > 0 else 1
    else:
        _cur = _cur - 1 if _cur < 0 else -1
    _streaks[_i] = _cur
data["Streak"] = _streaks

# ── Day of week ───────────────────────────────────────────────
data["Day_of_week"] = data.index.dayofweek

# ── Lag features ─────────────────────────────────────────────
for lag in [1, 2, 3, 5]:
    data[f"Return_lag{lag}"]    = data["Return_1d"].shift(lag)
    data[f"Vol_change_lag{lag}"] = data["Vol_change"].shift(lag)
    data[f"RSI_lag{lag}"]       = data["RSI"].shift(lag)


# ============================================================
# PHASE 3: CROSS-MARKET FEATURES
# ============================================================
print("[3] Adding cross-market signals...")

es_dates = data.index

for name in ["VIX", "TNX", "DXY", "NQ", "GC", "CL"]:
    if name not in all_data:
        continue
    other = all_data[name]["Close"].reindex(es_dates, method="ffill")
    data[f"{name}_return"]  = other.pct_change()
    data[f"{name}_mom5"]    = other.pct_change(5)
    data[f"{name}_corr20"]  = data["Return_1d"].rolling(20).corr(other.pct_change())

if "VIX" in all_data:
    vix = all_data["VIX"]["Close"].reindex(es_dates, method="ffill")
    data["VIX_level"]    = vix
    data["VIX_MA10"]     = vix.rolling(10).mean()
    data["VIX_vs_MA10"]  = vix / vix.rolling(10).mean() - 1   # VIX spike vs its own MA
    data["VIX_high"]     = (vix > 25).astype(int)
    data["VIX_extreme"]  = (vix > 30).astype(int)
    data["VIX_change5d"] = vix.pct_change(5)

if "TNX" in all_data:
    tnx = all_data["TNX"]["Close"].reindex(es_dates, method="ffill")
    data["TNX_level"]    = tnx
    data["TNX_change1d"] = tnx.pct_change(1)
    data["TNX_change5d"] = tnx.pct_change(5)

if "DXY" in all_data:
    dxy = all_data["DXY"]["Close"].reindex(es_dates, method="ffill")
    data["DXY_vs_MA20"]  = dxy / dxy.rolling(20).mean() - 1

if "NQ" in all_data:
    nq = all_data["NQ"]["Close"].reindex(es_dates, method="ffill")
    data["ES_NQ_spread"] = data["Return_1d"] - nq.pct_change()
    data["NQ_mom5"]      = nq.pct_change(5)

if "GC" in all_data:
    data["ES_GC_spread"] = data["Return_1d"] - all_data["GC"]["Close"].reindex(es_dates, method="ffill").pct_change()

if "CL" in all_data:
    data["CL_mom5"] = all_data["CL"]["Close"].reindex(es_dates, method="ffill").pct_change(5)


# ============================================================
# PHASE 4: REGIME DETECTION + INTERACTION FEATURES
# ============================================================
print("[4] Detecting regimes + building interaction features...")

# Regime 1: Volatility (calm=0 / normal=1 / turbulent=2)
vol_20    = data["Volatility_20"]
vol_q25   = vol_20.rolling(60).quantile(0.25)
vol_q75   = vol_20.rolling(60).quantile(0.75)
data["Vol_regime"] = 1
data.loc[vol_20 < vol_q25, "Vol_regime"] = 0
data.loc[vol_20 > vol_q75, "Vol_regime"] = 2

# Regime 2: Trend (downtrend=0 / range=1 / uptrend=2)
ma50_slope = data["MA_50"].pct_change(10)
data["Trend_regime"] = 1
data.loc[(data["Price_vs_MA50"] >  0.02) & (ma50_slope > 0), "Trend_regime"] = 2
data.loc[(data["Price_vs_MA50"] < -0.02) & (ma50_slope < 0), "Trend_regime"] = 0

# Regime 3: VIX (complacent=0 / normal=1 / fear=2 / panic=3)
if "VIX" in all_data:
    _vix = vix.reindex(data.index, method="ffill")
    data["VIX_regime"] = 1
    data.loc[_vix < 15, "VIX_regime"] = 0
    data.loc[_vix > 25, "VIX_regime"] = 2
    data.loc[_vix > 30, "VIX_regime"] = 3

# Regime 4: Correlation (rotation=0 / normal=1 / herd=2)
if "NQ" in all_data and "GC" in all_data:
    data["Corr_regime"] = 1
    _nq_corr = data.get("NQ_corr20", pd.Series(0, index=data.index))
    data.loc[_nq_corr > 0.7, "Corr_regime"] = 2
    data.loc[_nq_corr < 0.3, "Corr_regime"] = 0

# Combined regime label (for per-regime accuracy analysis)
data["Regime_label"] = data["Trend_regime"].astype(str) + "_" + data["Vol_regime"].astype(str)

# ── Regime interaction features ───────────────────────────────
# These teach the model that the same RSI/MACD value means different
# things in different trend and volatility contexts
data["RSI_x_downtrend"] = data["RSI"] * (data["Trend_regime"] == 0).astype(int)
data["RSI_x_uptrend"]   = data["RSI"] * (data["Trend_regime"] == 2).astype(int)
data["Mom5_x_vol"]      = data["Return_5d"] * data["Vol_regime"]
data["BB_x_trend"]      = data["BB_position"] * data["Trend_regime"]
data["Stoch_x_trend"]   = data["Stoch_K"] * data["Trend_regime"]
if "VIX_regime" in data.columns:
    data["MACD_x_vix"]  = data["MACD_hist"] * data["VIX_regime"]

regime_features = ["Vol_regime", "Trend_regime"]
if "VIX_regime"  in data.columns: regime_features.append("VIX_regime")
if "Corr_regime" in data.columns: regime_features.append("Corr_regime")

print(f"  Regime types: {data['Regime_label'].nunique()} distinct")
print(f"  Current regime: Trend={data['Trend_regime'].iloc[-1]} "
      f"({'Downtrend' if data['Trend_regime'].iloc[-1]==0 else 'Range' if data['Trend_regime'].iloc[-1]==1 else 'Uptrend'}), "
      f"Vol={data['Vol_regime'].iloc[-1]} "
      f"({'Calm' if data['Vol_regime'].iloc[-1]==0 else 'Normal' if data['Vol_regime'].iloc[-1]==1 else 'Turbulent'})")


# ============================================================
# PHASE 5: MULTI-HORIZON TARGET
# ============================================================
# Majority vote across 3 horizons → reduces single-day noise
# We still trade on 1-day returns; this is the training LABEL
data["Target_1d"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data["Target_3d"] = (data["Close"].shift(-3) > data["Close"]).astype(int)
data["Target_5d"] = (data["Close"].shift(-5) > data["Close"]).astype(int)
data["Target"]    = ((data["Target_1d"] + data["Target_3d"] + data["Target_5d"]) >= 2).astype(int)


# ============================================================
# PHASE 6: CLEAN
# ============================================================
data = data.replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n[6] Clean data: {len(data)} rows | Up label: {data['Target'].mean():.1%}")


# ============================================================
# PHASE 7: FEATURE SELECTION — WARMUP WINDOW ONLY (NO LEAK)
# ============================================================
# v3 BUG: MI was run on all data, leaking future info into feature selection.
# Fix: run on the first training window only.
print("\n[7] Feature selection on warmup window (no look-ahead)...")

TRAIN_SIZE = 250   # ~1 year per fold
TEST_SIZE  = 40    # ~2 months per fold

EXCLUDE = {
    "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits",
    "Target", "Target_1d", "Target_3d", "Target_5d",
    "MA_5", "MA_10", "MA_20", "MA_50", "MA_100",
    "Vol_MA5", "Vol_MA20", "VIX_level", "VIX_MA10", "TNX_level",
    "ATR_14", "Regime_label", "OBV",
}
all_feature_cols = [c for c in data.columns if c not in EXCLUDE]

# Feature selection strictly on the first TRAIN_SIZE rows
X_warmup = data[all_feature_cols].iloc[:TRAIN_SIZE].values
y_warmup = data["Target"].iloc[:TRAIN_SIZE].values

mi_scores  = mutual_info_classif(X_warmup, y_warmup, random_state=42)
mi_ranking = sorted(zip(all_feature_cols, mi_scores), key=lambda x: x[1], reverse=True)

N_FEATURES = 25
top_features = [f[0] for f in mi_ranking[:N_FEATURES]]

print(f"  Reduced: {len(all_feature_cols)} → {N_FEATURES} features")
print(f"\n  TOP {N_FEATURES} FEATURES (mutual info on warmup window):")
print(f"  {'—' * 52}")
for i, (feat, score) in enumerate(mi_ranking[:N_FEATURES]):
    bar = "█" * int(score * 300)
    print(f"  {i+1:>3}. {feat:<30} {score:.4f}  {bar}")

feature_cols = top_features
X     = data[feature_cols].values
y     = data["Target"].values
dates = data.index


# ============================================================
# PHASE 8: ADAPTIVE WALK-FORWARD — ENSEMBLE
# ============================================================
print(f"\n{'=' * 62}")
print(f"  WALK-FORWARD ENSEMBLE  [{' + '.join(_ensemble_names)}]")
print(f"{'=' * 62}")

all_predictions  = []
all_actuals      = []
all_dates_list   = []
all_returns      = []
all_confidences  = []
all_regimes      = []

# Track per-model rolling accuracy to see who contributes most
model_fold_accs  = {m: [] for m in _ensemble_names}

# Save last fold's models + scaler for the live signal
last_models  = {}
last_scaler  = None

step = 0
i    = 0
while i + TRAIN_SIZE + TEST_SIZE <= len(X):
    X_train = X[i : i + TRAIN_SIZE]
    y_train = y[i : i + TRAIN_SIZE]
    X_test  = X[i + TRAIN_SIZE : i + TRAIN_SIZE + TEST_SIZE]
    y_test  = y[i + TRAIN_SIZE : i + TRAIN_SIZE + TEST_SIZE]

    test_dates   = dates[i + TRAIN_SIZE : i + TRAIN_SIZE + TEST_SIZE]
    test_returns = data["Return_1d"].iloc[i + TRAIN_SIZE : i + TRAIN_SIZE + TEST_SIZE].values
    test_regimes = data["Regime_label"].iloc[i + TRAIN_SIZE : i + TRAIN_SIZE + TEST_SIZE].values

    # Scale per-fold (prevents future data from influencing scaling)
    scaler_fold  = StandardScaler()
    X_train_s    = scaler_fold.fit_transform(X_train)
    X_test_s     = scaler_fold.transform(X_test)

    # ── Train all models ─────────────────────────────────────
    fold_models  = {}
    probs_list   = []

    gbm = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    gbm.fit(X_train_s, y_train)
    gbm_probs = gbm.predict_proba(X_test_s)[:, 1]
    fold_models["GBM"] = gbm
    probs_list.append(gbm_probs)
    model_fold_accs["GBM"].append(accuracy_score(y_test, (gbm_probs > 0.5).astype(int)))

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=20,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train_s, y_train)
    rf_probs = rf.predict_proba(X_test_s)[:, 1]
    fold_models["RF"] = rf
    probs_list.append(rf_probs)
    model_fold_accs["RF"].append(accuracy_score(y_test, (rf_probs > 0.5).astype(int)))

    if HAS_LGB:
        lgbm = lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_child_samples=20, verbose=-1, random_state=42
        )
        lgbm.fit(X_train_s, y_train)
        lgb_probs = lgbm.predict_proba(X_test_s)[:, 1]
        fold_models["LGB"] = lgbm
        probs_list.append(lgb_probs)
        model_fold_accs["LGB"].append(accuracy_score(y_test, (lgb_probs > 0.5).astype(int)))

    if HAS_XGB:
        xgbm = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_child_weight=20, verbosity=0,
            eval_metric='logloss', random_state=42
        )
        xgbm.fit(X_train_s, y_train)
        xgb_probs = xgbm.predict_proba(X_test_s)[:, 1]
        fold_models["XGB"] = xgbm
        probs_list.append(xgb_probs)
        model_fold_accs["XGB"].append(accuracy_score(y_test, (xgb_probs > 0.5).astype(int)))

    # ── Ensemble: average probabilities ──────────────────────
    ensemble_probs = np.mean(probs_list, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    acc            = accuracy_score(y_test, ensemble_preds)

    last_models = fold_models
    last_scaler = scaler_fold

    step        += 1
    dom_regime   = pd.Series(test_regimes).mode().iloc[0]
    period_start = test_dates[0].strftime("%Y-%m-%d")
    period_end   = test_dates[-1].strftime("%Y-%m-%d")
    print(f"  [{step:>2}] {period_start} → {period_end}  Acc: {acc:.1%}  Regime: {dom_regime}")

    all_predictions.extend(ensemble_preds)
    all_actuals.extend(y_test)
    all_dates_list.extend(test_dates)
    all_returns.extend(test_returns)
    all_confidences.extend(ensemble_probs)
    all_regimes.extend(test_regimes)

    i += TEST_SIZE

all_predictions  = np.array(all_predictions)
all_actuals      = np.array(all_actuals)
all_returns      = np.array(all_returns)
all_confidences  = np.array(all_confidences)
all_regimes      = np.array(all_regimes)

overall_acc = accuracy_score(all_actuals, all_predictions)
print(f"\n  ENSEMBLE ACCURACY: {overall_acc:.2%}  ({len(all_predictions)} predictions)")
print(f"\n  Per-model avg accuracy across {step} windows:")
for mname, accs in model_fold_accs.items():
    indicator = "✓" if np.mean(accs) > 0.53 else "~"
    print(f"    {indicator} {mname:>4}: {np.mean(accs):.2%}")


# ============================================================
# PHASE 9: REGIME-SPECIFIC ACCURACY
# ============================================================
print(f"\n{'=' * 62}")
print("  ACCURACY BY MARKET REGIME")
print(f"{'=' * 62}")

REGIME_NAMES = {
    "0_0": "Downtrend + Calm",        "0_1": "Downtrend + Normal Vol",
    "0_2": "Downtrend + Turbulent",   "1_0": "Range + Calm",
    "1_1": "Range + Normal Vol",      "1_2": "Range + Turbulent",
    "2_0": "Uptrend + Calm",          "2_1": "Uptrend + Normal Vol",
    "2_2": "Uptrend + Turbulent",
}

for regime in sorted(np.unique(all_regimes)):
    mask  = all_regimes == regime
    if sum(mask) < 5:
        continue
    racc  = accuracy_score(all_actuals[mask], all_predictions[mask])
    rname = REGIME_NAMES.get(regime, regime)
    flag  = "✓" if racc > 0.53 else ("✗" if racc < 0.48 else "~")
    print(f"  {flag} {rname:<30} {racc:.1%}  ({sum(mask)} days)")


# ============================================================
# PHASE 10: CONFIDENCE-FILTERED BACKTEST
# ============================================================
print(f"\n{'=' * 62}")
print("  CONFIDENCE-FILTERED BACKTEST")
print(f"{'=' * 62}")
print(f"  {'Threshold':>10} {'Trades':>7} {'Skip%':>6} {'WinRate':>8} {'ProfFac':>8} {'Strategy':>9} {'B&H':>8}")
print(f"  {'—'*65}")

for thresh in [0.50, 0.52, 0.55, 0.58, 0.60]:
    long_sig = all_confidences > thresh
    filt     = np.where(long_sig, 1, 0)
    s_rets   = all_returns[1:] * filt[:-1]
    trades   = sum(filt[:-1])
    if trades < 10:
        continue

    strat  = (1 + s_rets).prod() - 1
    bh     = (1 + all_returns[1:]).prod() - 1
    wins   = sum((s_rets > 0) & (filt[:-1] == 1))
    losses = sum((s_rets < 0) & (filt[:-1] == 1))
    wr     = wins / trades * 100
    avg_w  = s_rets[(s_rets > 0)  & (filt[:-1] == 1)].mean() if wins   > 0 else 0
    avg_l  = abs(s_rets[(s_rets < 0) & (filt[:-1] == 1)].mean()) if losses > 0 else 0
    pf     = avg_w / avg_l if avg_l > 0 else 0
    skip   = (1 - trades / len(filt)) * 100
    beat   = "▲" if strat > bh else "▼"

    print(f"  {thresh:.0%} conf  {trades:>6}  {skip:>5.0f}%  {wr:>7.1f}%  {pf:>7.2f}  "
          f"  {strat:>8.2%}  {bh:>7.2%} {beat}")


# ============================================================
# PHASE 11: RISK-MANAGED BACKTEST v4
#   Kelly criterion | 0.01% txn costs | drawdown guard
# ============================================================
print(f"\n{'=' * 62}")
print("  RISK-MANAGED BACKTEST v4")
print(f"  Kelly fraction: 25% | Txn cost: 0.01%/side | DD guard: 10%")
print(f"{'=' * 62}")

TXN_COST     = 0.0001    # 0.01% per side → 0.02% round trip
DD_GUARD     = 0.10      # halve size when drawdown exceeds this
KELLY_FRAC   = 0.25      # conservative fractional Kelly
KELLY_WINDOW = 60        # rolling window of past trades to estimate Kelly

def fractional_kelly(past_returns: np.ndarray, fraction: float = KELLY_FRAC) -> float:
    """
    Estimate fractional Kelly multiplier from a window of past trade returns.
    Returns a multiplier in [0.1, 3.0].
    """
    wins  = past_returns[past_returns > 0]
    loses = past_returns[past_returns < 0]
    if len(wins) == 0 or len(loses) == 0:
        return 1.0
    p     = len(wins) / len(past_returns)
    b     = wins.mean() / abs(loses.mean())       # reward-to-risk ratio
    kelly = (p * (b + 1) - 1) / b
    return float(np.clip(kelly * fraction, 0.1, 3.0))

capital    = 100_000
equity     = [float(capital)]
trade_log  = []

for t in range(len(all_predictions) - 1):
    pred    = all_predictions[t]
    conf    = all_confidences[t]
    ret_nxt = all_returns[t + 1]

    if pred == 0:
        equity.append(equity[-1])
        continue

    # ── Confidence multiplier: 1.0x @ 50%, up to 2.0x @ 70%+ ──
    conf_mult = 1.0 + min(conf - 0.5, 0.20) * 10.0   # linear ramp

    # ── Kelly from recent trade history ──────────────────────
    if len(trade_log) >= 10:
        past  = np.array([tl["return"] for tl in trade_log[-KELLY_WINDOW:]])
        kelly = fractional_kelly(past)
    else:
        kelly = 1.0

    size = conf_mult * kelly

    # ── Regime risk filter: halve in turbulent regimes ───────
    if all_regimes[t].split("_")[1] == "2":
        size *= 0.5

    # ── Drawdown guard: halve when equity is > DD_GUARD below peak ──
    peak = max(equity)
    if (equity[-1] - peak) / peak < -DD_GUARD:
        size *= 0.5

    size = float(np.clip(size, 0.1, 3.0))

    # ── Apply trade: gross return → minus round-trip costs ───
    trade_gross = ret_nxt * size
    trade_net   = trade_gross - 2 * TXN_COST * size
    # Hard stop: cap loss at 1.5% per unit of size
    if trade_net < -0.015 * size:
        trade_net = -0.015 * size

    equity.append(equity[-1] * (1 + trade_net))
    trade_log.append({"return": trade_net, "gross": trade_gross,
                      "conf": conf, "regime": all_regimes[t], "size": size})

equity = np.array(equity)
total_ret  = (equity[-1] / equity[0]) - 1
peak_eq    = np.maximum.accumulate(equity)
drawdowns  = (equity - peak_eq) / peak_eq
max_dd     = drawdowns.min()

# Buy & hold benchmark
bh_equity = np.cumprod(np.r_[capital, capital * (1 + all_returns[1:])])
bh_ret    = (bh_equity[-1] / bh_equity[0]) - 1
bh_dd     = ((bh_equity - np.maximum.accumulate(bh_equity)) / np.maximum.accumulate(bh_equity)).min()

trade_rets = np.array([t["return"] for t in trade_log])
wins_r     = trade_rets[trade_rets > 0]
loss_r     = trade_rets[trade_rets < 0]
avg_win_v  = wins_r.mean()       if len(wins_r) > 0 else 0
avg_los_v  = abs(loss_r.mean())  if len(loss_r) > 0 else 0
pf_v4      = avg_win_v / avg_los_v if avg_los_v > 0 else 0
winrate_v4 = len(wins_r) / len(trade_rets) * 100 if len(trade_rets) > 0 else 0

daily_rets = np.diff(equity) / equity[:-1]
std_all    = np.std(daily_rets)
std_down   = np.std(daily_rets[daily_rets < 0]) if any(daily_rets < 0) else 1e-9
sharpe     = np.mean(daily_rets) / std_all  * np.sqrt(252) if std_all  > 0 else 0
sortino    = np.mean(daily_rets) / std_down * np.sqrt(252) if std_down > 0 else 0

n_years    = len(equity) / 252
ann_ret    = (equity[-1] / equity[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
calmar     = ann_ret / abs(max_dd) if max_dd != 0 else 0

print(f"""
  Starting capital:  ${capital:>12,.0f}
  Final equity:      ${equity[-1]:>12,.0f}
  Total return:      {total_ret:>10.2%}
  Buy & hold:        {bh_ret:>10.2%}

  Total trades:      {len(trade_rets):>6}
  Win rate:          {winrate_v4:>8.1f}%
  Avg win:           {avg_win_v:>10.4f}
  Avg loss:          {avg_los_v:>10.4f}
  Profit factor:     {pf_v4:>10.2f}

  Max drawdown:      {max_dd:>10.2%}   (B&H: {bh_dd:.2%})
  Sharpe ratio:      {sharpe:>10.2f}
  Sortino ratio:     {sortino:>10.2f}
  Calmar ratio:      {calmar:>10.2f}

  {'▲ MODEL BEATS BUY & HOLD' if total_ret > bh_ret else '▼ BUY & HOLD STILL WINS'}   Δ = {total_ret - bh_ret:+.2%}
""")


# ============================================================
# PHASE 12: LSTM (requires TensorFlow)
# ============================================================
print(f"{'=' * 62}")
print("  LSTM — REGIME-AWARE  (requires TensorFlow)")
print(f"{'=' * 62}")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    SEQ_LEN     = 10
    scaler_lstm = StandardScaler()
    X_scaled    = scaler_lstm.fit_transform(X)

    X_seq, y_seq = [], []
    for j in range(SEQ_LEN, len(X_scaled)):
        X_seq.append(X_scaled[j - SEQ_LEN : j])
        y_seq.append(y[j])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split         = int(len(X_seq) * 0.80)
    X_tr, X_te    = X_seq[:split], X_seq[split:]
    y_tr, y_te    = y_seq[:split], y_seq[split:]

    print(f"\n  Train: {len(X_tr)} sequences  |  Test: {len(X_te)}  |  Features/step: {X_seq.shape[2]}")
    print(f"  Training...")

    model_lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])),
        BatchNormalization(), Dropout(0.30),
        LSTM(32),
        BatchNormalization(), Dropout(0.30),
        Dense(16, activation="relu"), Dropout(0.20),
        Dense(1,  activation="sigmoid"),
    ])
    model_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    cbs = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
    ]
    model_lstm.fit(X_tr, y_tr, epochs=100, batch_size=32,
                   validation_split=0.15, callbacks=cbs, verbose=0)

    lstm_probs = model_lstm.predict(X_te, verbose=0).flatten()
    lstm_preds = (lstm_probs > 0.5).astype(int)
    lstm_acc   = accuracy_score(y_te, lstm_preds)
    print(f"\n  LSTM Accuracy: {lstm_acc:.2%}")

    for ct in [0.55, 0.60]:
        mask = (lstm_probs > ct) | (lstm_probs < (1 - ct))
        if sum(mask) > 10:
            print(f"  LSTM @ {ct:.0%} conf: {accuracy_score(y_te[mask], lstm_preds[mask]):.2%}  ({sum(mask)} trades)")

    # LSTM backtest (confidence-sized, stop-lossed)
    te_rets = data["Return_1d"].iloc[split + SEQ_LEN : split + SEQ_LEN + len(lstm_preds)].values
    if len(te_rets) >= len(lstm_preds):
        te_rets  = te_rets[:len(lstm_preds)]
        lsizes   = np.where(lstm_probs > 0.60, 2.0,
                   np.where(lstm_probs > 0.55, 1.5,
                   np.where(lstm_probs > 0.50, 1.0, 0.0)))
        l_strat  = np.clip(te_rets[1:] * lsizes[:-1], -0.015, None)
        l_tot    = (1 + l_strat).prod() - 1
        l_bh     = (1 + te_rets[1:]).prod() - 1
        l_tr     = sum(lsizes[:-1] > 0)
        l_wr     = sum((l_strat > 0) & (lsizes[:-1] > 0)) / l_tr * 100 if l_tr > 0 else 0
        print(f"\n  LSTM backtest: {l_tot:.2%} vs B&H {l_bh:.2%}  |  Trades: {l_tr}  Win: {l_wr:.1f}%")

except ImportError:
    print("\n  TensorFlow not installed — skipping LSTM.")
    print("  Install with:  py -m pip install tensorflow")
    lstm_acc = 0


# ============================================================
# PHASE 13: LIVE SIGNAL — NEXT TRADING DAY
# ============================================================
print(f"\n{'=' * 62}")
print("  LIVE SIGNAL — NEXT TRADING DAY")
print(f"{'=' * 62}")

if last_models and last_scaler is not None:
    last_row   = data[feature_cols].iloc[-1:].values
    last_row_s = last_scaler.transform(last_row)

    live_probs = {name: model.predict_proba(last_row_s)[0][1]
                  for name, model in last_models.items()}
    live_prob  = float(np.mean(list(live_probs.values())))
    live_conf  = max(live_prob, 1 - live_prob)
    live_dir   = "📈 UP" if live_prob > 0.5 else "📉 DOWN"

    cur_trend = int(data["Trend_regime"].iloc[-1])
    cur_vol   = int(data["Vol_regime"].iloc[-1])
    trend_str = ["Downtrend", "Ranging", "Uptrend"][cur_trend]
    vol_str   = ["Calm", "Normal", "Turbulent"][cur_vol]

    print(f"\n  Direction:     {live_dir}")
    print(f"  Confidence:    {live_conf:.1%}")
    print(f"  Market regime: {trend_str} + {vol_str}")
    print(f"  Data through:  {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"\n  Per-model:     "
          + "  |  ".join(f"{k}: {v:.1%}" for k, v in live_probs.items()))
    print()


# ============================================================
# FINAL VERDICT
# ============================================================
print(f"{'=' * 62}")
print("  FINAL VERDICT — v4 ENSEMBLE")
print(f"{'=' * 62}")
print(f"""
  v4 IMPROVEMENTS OVER v3:
  ✓ Feature selection leak fixed (warmup window, not all data)
  ✓ Multi-horizon target (1d + 3d + 5d majority vote)
  ✓ New features: ROC, Stochastic, OBV, 52w levels, gap, regime interactions
  ✓ Ensemble: {' + '.join(_ensemble_names)} (averaged probabilities)
  ✓ Kelly criterion position sizing (fractional 25%)
  ✓ Transaction costs (0.01% per side, round-trip deducted)
  ✓ Drawdown guard (50% size cut when DD > {DD_GUARD:.0%})
  ✓ Additional metrics: Sortino + Calmar
  ✓ Live next-day signal

  Walk-forward accuracy:   {overall_acc:.2%}
  Risk-managed return:     {total_ret:.2%}
  Buy & hold return:       {bh_ret:.2%}
  Outperformance:          {total_ret - bh_ret:+.2%}
  Max drawdown:            {max_dd:.2%}
  Sharpe ratio:            {sharpe:.2f}
  Sortino ratio:           {sortino:.2f}
  Calmar ratio:            {calmar:.2f}
  Profit factor:           {pf_v4:.2f}
""")
