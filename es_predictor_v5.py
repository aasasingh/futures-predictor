"""
ES Futures — two engines:
  Engine 1: next-day direction (logistic regression, ~11 features)
  Engine 2: 5-day vol expansion (logistic regression, same features)

Strategy output: vol-targeted sizing (w = TARGET_VOL / predicted_vol).
Direction model is kept for reference; vol engine is the actionable signal.
"""
import warnings, os, sys, datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAIN_SIZE = 250    # ~1 year
TEST_SIZE  = 40     # ~2 months per fold
PURGE      = 5      # rows dropped from train tail (vol target looks 5d forward)
HOLDOUT    = 126    # final ~6 months, never seen during walk-forward
TARGET_VOL = 0.01   # 1% daily vol target for vol-sized strategy
TXN_COST   = 0.0001 # one-way; applied twice per round-trip

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
print("=" * 60)
print("Data")
print("=" * 60)

TICKERS = {
    "ES": "ES=F", "VIX": "^VIX", "VIX3M": "^VIX3M",
    "TNX": "^TNX", "HYG": "HYG",
    "XLK": "XLK", "XLF": "XLF", "XLE": "XLE",
    "XLV": "XLV", "XLI": "XLI", "XLU": "XLU",
}

# Only completed sessions; futures print a bar for today while still trading
today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

raw = {}
for name, ticker in TICKERS.items():
    try:
        d = yf.Ticker(ticker).history(period="5y")
        # ^VIX/^TNX are in Chicago time, the rest New York. Align on plain dates
        # or the reindex below matches them to the previous day.
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d.index = d.index.normalize()
        d = d[d.index < today]
        if len(d) > 100:
            raw[name] = d
            print(f"  {name:>5}: {len(d)} rows")
        else:
            print(f"  {name:>5}: too short ({len(d)} rows)")
    except Exception as e:
        print(f"  {name:>5}: {e}")

data = raw["ES"].copy()
idx  = data.index

fred_data = {}
FRED_KEY = os.environ.get("FRED_API_KEY")
if HAS_FRED and FRED_KEY:
    fred     = Fred(api_key=FRED_KEY)
    for sid in ["T10Y2Y", "BAMLH0A0HYM2"]:
        try:
            s = fred.get_series(sid, observation_start="2020-01-01")
            s.index = pd.to_datetime(s.index)
            s = s.reindex(idx, method="ffill")
            # FRED only serves 3 years of ICE BofA series like HY OAS; a short
            # series would make dropna() discard most of the history.
            # Skipping it falls back to HYG below.
            if s.notna().mean() < 0.9:
                print(f"  FRED {sid}: only {s.notna().sum()} rows, skipped")
                continue
            fred_data[sid] = s
            print(f"  FRED {sid}: {s.notna().sum()} valid rows")
        except Exception as e:
            print(f"  FRED {sid}: {e}")
else:
    print("  FRED: set FRED_API_KEY for macro features")

# ---------------------------------------------------------------------------
# FEATURES
# Fixed set, chosen before seeing results, one rationale per feature.
# Built from data up to day t, then shifted 1 day so row t only sees t-1.
# f_raw keeps the unshifted version; its last row is the live-signal input.
# ---------------------------------------------------------------------------
print("\nFeatures")

ret   = data["Close"].pct_change()
vol5  = ret.rolling(5).std()
vol20 = ret.rolling(20).std()
delta = data["Close"].diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi14 = 100 - 100 / (1 + gain / loss)

f = pd.DataFrame(index=idx)

# Short-term return autocorrelation / reversal
f["ret_1d"]    = ret
f["ret_5d"]    = ret.rolling(5).sum()

# Vol clustering: today's vol predicts tomorrow's
f["vol_20"]    = vol20
f["vol_ratio"] = vol5 / vol20   # >1: short-term vol elevated

# Mean reversion
f["rsi"]       = rsi14

# Implied vol; fear gauge and expectations about near-term vol
if "VIX" in raw:
    vix = raw["VIX"]["Close"].reindex(idx, method="ffill")
    f["vix"] = vix
    if "VIX3M" in raw:
        vix3m = raw["VIX3M"]["Close"].reindex(idx, method="ffill")
        f["vix_term"] = vix / vix3m  # >1: backwardation / acute fear

# Rising rates compress equity multiples
if "TNX" in raw:
    tnx = raw["TNX"]["Close"].reindex(idx, method="ffill")
    f["tnx_chg5"] = tnx.pct_change(5)

# Macro regime: inversion historically precedes contractions
if "T10Y2Y" in fred_data:
    f["yield_curve"] = fred_data["T10Y2Y"]

# Systemic risk appetite; credit leads equity at turning points
if "BAMLH0A0HYM2" in fred_data:
    f["hy_oas"] = fred_data["BAMLH0A0HYM2"]
elif "HYG" in raw:
    hyg = raw["HYG"]["Close"].reindex(idx, method="ffill")
    f["hyg_mom5"] = hyg.pct_change(5)

# Breadth: % of sectors with positive 20d return
sectors = [s for s in ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU"] if s in raw]
if len(sectors) >= 4:
    smom = pd.DataFrame({
        s: raw[s]["Close"].reindex(idx, method="ffill").pct_change(20)
        for s in sectors
    })
    f["breadth"] = (smom > 0).sum(axis=1) / len(sectors)

f_raw = f.replace([np.inf, -np.inf], np.nan)
f = f_raw.shift(1)

feat_cols = list(f.columns)
print(f"  {len(feat_cols)} features: {feat_cols}")

# ---------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------
y_dir_raw = (ret > 0).astype(int)                         # Engine 1: direction
y_vol_raw = (vol5.shift(-5) > vol5).astype(int)           # Engine 2: vol expansion in 5d
# NaN > x is False, so the last 5 rows were labelled 0; mark them unknown
y_vol_raw = y_vol_raw.where(vol5.shift(-5).notna())

# Persistence benchmark for vol: is vol currently above long-term average?
# (tests whether the model adds value over "just look at current vol regime")
persist_raw = (f["vol_ratio"] > 1).astype(int) if "vol_ratio" in f.columns \
              else pd.Series(0, index=idx)

df = pd.concat([
    f,
    y_dir_raw.rename("y_dir"),
    y_vol_raw.rename("y_vol"),
    persist_raw.rename("y_persist"),
    ret.rename("ret"),
    vol20.shift(1).rename("vol20_prev"),   # vol known before day t, for sizing
], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
df[["y_dir", "y_vol"]] = df[["y_dir", "y_vol"]].astype(int)

up_rate  = df["y_dir"].mean()
vol_rate = df["y_vol"].mean()
pers_acc_full = accuracy_score(df["y_vol"], df["y_persist"])

print(f"\n  Clean rows: {len(df)}")
print(f"  Direction up-rate:    {up_rate:.1%}  "
      f"(majority-class baseline: {max(up_rate, 1-up_rate):.1%})")
print(f"  Vol-expansion rate:   {vol_rate:.1%}  "
      f"(majority-class baseline: {max(vol_rate, 1-vol_rate):.1%}  "
      f"persistence: {pers_acc_full:.1%})")

X_all       = df[feat_cols].values
y_dir_all   = df["y_dir"].values
y_vol_all   = df["y_vol"].values
persist_all = df["y_persist"].values
dates       = df.index
rets_all    = df["ret"].values
vol20_all   = df["vol20_prev"].values

# ---------------------------------------------------------------------------
# WALK-FORWARD CV
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Walk-forward  train={TRAIN_SIZE}  test={TEST_SIZE}  purge={PURGE}  holdout={HOLDOUT}")
print("=" * 60)

# Reserve holdout — never touched until final evaluation
ho_X     = X_all[-HOLDOUT:]
ho_y_dir = y_dir_all[-HOLDOUT:]
ho_y_vol = y_vol_all[-HOLDOUT:]
ho_dates = dates[-HOLDOUT:]
ho_rets  = rets_all[-HOLDOUT:]
ho_vol20 = vol20_all[-HOLDOUT:]
ho_pers  = persist_all[-HOLDOUT:]

X_wf     = X_all[:-HOLDOUT]
y_wf_dir = y_dir_all[:-HOLDOUT]
y_wf_vol = y_vol_all[:-HOLDOUT]
d_wf     = dates[:-HOLDOUT]
r_wf     = rets_all[:-HOLDOUT]
v_wf     = vol20_all[:-HOLDOUT]
pers_wf  = persist_all[:-HOLDOUT]

# Accumulators
res = {k: [] for k in [
    "preds_dir", "probs_dir", "preds_vol", "probs_vol",
    "actuals_dir", "actuals_vol", "rets", "vol20",
    "persist", "base_dir", "base_vol",
]}

last_lr_dir = last_lr_vol = last_scaler = None

print(f"  {'fold':>4}  {'date':>10}  {'dir':>7}  {'dir_base':>9}  "
      f"{'vol':>7}  {'vol_base':>9}  {'persist':>8}")

i = step = 0
while i + TRAIN_SIZE + PURGE + TEST_SIZE <= len(X_wf):
    # Drop last PURGE rows from training: vol target y_vol = vol5.shift(-5) > vol5
    # means the last 5 rows of any training window contain returns from the test window.
    tr = slice(i, i + TRAIN_SIZE - PURGE)
    te = slice(i + TRAIN_SIZE, i + TRAIN_SIZE + TEST_SIZE)

    X_tr     = X_wf[tr];     X_te     = X_wf[te]
    y_tr_dir = y_wf_dir[tr]; y_te_dir = y_wf_dir[te]
    y_tr_vol = y_wf_vol[tr]; y_te_vol = y_wf_vol[te]
    te_rets  = r_wf[te];     te_vol20 = v_wf[te]
    te_dates = d_wf[te];     te_pers  = pers_wf[te]

    fold_base_dir = y_tr_dir.mean()
    fold_base_vol = y_tr_vol.mean()

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    lr_dir  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_dir.fit(X_tr_s, y_tr_dir)
    p_dir   = lr_dir.predict_proba(X_te_s)[:, 1]

    lr_vol  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_vol.fit(X_tr_s, y_tr_vol)
    p_vol   = lr_vol.predict_proba(X_te_s)[:, 1]

    acc_dir  = accuracy_score(y_te_dir, (p_dir > 0.5).astype(int))
    acc_vol  = accuracy_score(y_te_vol, (p_vol > 0.5).astype(int))
    acc_pers = accuracy_score(y_te_vol, te_pers)

    step += 1
    print(f"  {step:>4}  {te_dates[0].strftime('%Y-%m-%d')}  "
          f"{acc_dir:>6.1%}  {max(fold_base_dir,1-fold_base_dir):>8.1%}  "
          f"{acc_vol:>6.1%}  {max(fold_base_vol,1-fold_base_vol):>8.1%}  "
          f"{acc_pers:>7.1%}")

    n = len(y_te_dir)
    res["preds_dir"].extend((p_dir > 0.5).astype(int))
    res["probs_dir"].extend(p_dir)
    res["preds_vol"].extend((p_vol > 0.5).astype(int))
    res["probs_vol"].extend(p_vol)
    res["actuals_dir"].extend(y_te_dir)
    res["actuals_vol"].extend(y_te_vol)
    res["rets"].extend(te_rets)
    res["vol20"].extend(te_vol20)
    res["persist"].extend(te_pers)
    res["base_dir"].extend([fold_base_dir] * n)
    res["base_vol"].extend([fold_base_vol] * n)

    last_lr_dir = lr_dir
    last_lr_vol = lr_vol
    last_scaler = scaler
    i += TEST_SIZE

res = {k: np.array(v, dtype=float) for k, v in res.items()}

n_wf        = len(res["actuals_dir"])
acc_dir_ov  = accuracy_score(res["actuals_dir"].astype(int), res["preds_dir"].astype(int))
acc_vol_ov  = accuracy_score(res["actuals_vol"].astype(int), res["preds_vol"].astype(int))
acc_pers_ov = accuracy_score(res["actuals_vol"].astype(int), res["persist"].astype(int))
up_rate_wf  = res["actuals_dir"].mean()
vol_rate_wf = res["actuals_vol"].mean()

# Significance vs the majority-class rate, not 50% or the raw up-rate.
# If up-rate is 45%, "always down" scores 55%, so the bar is max(p, 1-p).
k_dir = int((res["preds_dir"].astype(int) == res["actuals_dir"].astype(int)).sum())
k_vol = int((res["preds_vol"].astype(int) == res["actuals_vol"].astype(int)).sum())
p_val_dir = stats.binomtest(k_dir, n_wf, max(up_rate_wf, 1 - up_rate_wf), alternative="greater").pvalue
p_val_vol = stats.binomtest(k_vol, n_wf, max(vol_rate_wf, 1 - vol_rate_wf), alternative="greater").pvalue

print(f"\n  Walk-forward summary ({n_wf} predictions, {step} folds)")
print(f"  Direction:     {acc_dir_ov:.2%}  "
      f"majority-class {max(up_rate_wf,1-up_rate_wf):.2%}  "
      f"(up-rate {up_rate_wf:.1%})  p={p_val_dir:.4f}")
print(f"  Vol expansion: {acc_vol_ov:.2%}  "
      f"majority-class {max(vol_rate_wf,1-vol_rate_wf):.2%}  "
      f"persistence {acc_pers_ov:.2%}  p={p_val_vol:.4f}")
print(f"  Direction {'significant' if p_val_dir < 0.05 else 'NOT significant'} vs base rate")
print(f"  Vol       {'significant' if p_val_vol < 0.05 else 'NOT significant'} vs base rate")

# ---------------------------------------------------------------------------
# BACKTESTS
# ---------------------------------------------------------------------------
def curve_stats(eq: np.ndarray):
    """Return (total_return, annualised_sharpe, max_drawdown)."""
    r  = eq[-1] / eq[0] - 1
    dr = np.diff(eq) / eq[:-1]
    sr = np.mean(dr) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0.0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    return r, sr, dd

print(f"\n{'='*60}")
print("Backtests")
print("=" * 60)

rets_wf   = res["rets"]
probs_dir = res["probs_dir"]
probs_vol = res["probs_vol"]
vol20_wf  = res["vol20"]

# probs[t] and vol20[t] only use data through t-1, so they trade day t's return

# Backtest A: direction model (long / short / flat)
eq_dir = [100_000.0]
for t in range(len(rets_wf)):
    r, p = rets_wf[t], probs_dir[t]
    if p > 0.55:
        eq_dir.append(eq_dir[-1] * (1 + r - 2 * TXN_COST))
    elif p < 0.45:
        eq_dir.append(eq_dir[-1] * (1 - r - 2 * TXN_COST))
    else:
        eq_dir.append(eq_dir[-1])
eq_dir = np.array(eq_dir)

# Backtest B: vol-targeted, always long
# size = (TARGET_VOL / current_vol20) * vol_adj
# vol_adj < 1 when model expects vol to expand (reduce size pre-emptively)
eq_volt = [100_000.0]
for t in range(len(rets_wf)):
    r    = rets_wf[t]
    v20  = max(vol20_wf[t], 1e-6)
    p_e  = probs_vol[t]
    vol_adj = 1.0 - (p_e - 0.5)         # [0.5, 1.5]; shrinks when expansion likely
    size    = float(np.clip(TARGET_VOL / v20 * vol_adj, 0.1, 3.0))
    eq_volt.append(eq_volt[-1] * (1 + r * size - 2 * TXN_COST * size))
eq_volt = np.array(eq_volt)

# B&H
eq_bh = 100_000.0 * np.cumprod(np.r_[1.0, 1 + rets_wf])

r_dir,  sr_dir,  dd_dir  = curve_stats(eq_dir)
r_volt, sr_volt, dd_volt = curve_stats(eq_volt)
r_bh,   sr_bh,   dd_bh   = curve_stats(eq_bh)

print(f"\n  {'Strategy':<22}  {'Return':>8}  {'Sharpe':>7}  {'MaxDD':>8}")
print(f"  {'Direction model':<22}  {r_dir:>8.2%}  {sr_dir:>7.2f}  {dd_dir:>8.2%}")
print(f"  {'Vol-targeted':<22}  {r_volt:>8.2%}  {sr_volt:>7.2f}  {dd_volt:>8.2%}")
print(f"  {'Buy & hold':<22}  {r_bh:>8.2%}  {sr_bh:>7.2f}  {dd_bh:>8.2%}")

# ---------------------------------------------------------------------------
# HOLDOUT
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Holdout  {ho_dates[0].strftime('%Y-%m-%d')} to {ho_dates[-1].strftime('%Y-%m-%d')}")
print("=" * 60)

if last_lr_dir is not None:
    ho_s       = last_scaler.transform(ho_X)
    ho_pd      = last_lr_dir.predict_proba(ho_s)[:, 1]
    ho_pv      = last_lr_vol.predict_proba(ho_s)[:, 1]

    ho_acc_dir  = accuracy_score(ho_y_dir, (ho_pd > 0.5).astype(int))
    ho_acc_vol  = accuracy_score(ho_y_vol, (ho_pv > 0.5).astype(int))
    ho_pers_acc = accuracy_score(ho_y_vol, ho_pers)
    ho_base_dir = ho_y_dir.mean()
    ho_base_vol = ho_y_vol.mean()

    print(f"  Direction:     {ho_acc_dir:.2%}  "
          f"majority-class {max(ho_base_dir,1-ho_base_dir):.2%}  "
          f"(up-rate {ho_base_dir:.1%})")
    print(f"  Vol expansion: {ho_acc_vol:.2%}  "
          f"majority-class {max(ho_base_vol,1-ho_base_vol):.2%}  "
          f"persistence {ho_pers_acc:.2%}")

    # Holdout backtest: vol-targeted
    ho_eq_volt = [100_000.0]
    ho_eq_bh   = [100_000.0]
    for t in range(len(ho_rets)):
        r    = ho_rets[t]
        v20  = max(ho_vol20[t], 1e-6)
        p_e  = ho_pv[t]
        vol_adj = 1.0 - (p_e - 0.5)
        size    = float(np.clip(TARGET_VOL / v20 * vol_adj, 0.1, 3.0))
        ho_eq_volt.append(ho_eq_volt[-1] * (1 + r * size - 2 * TXN_COST * size))
        ho_eq_bh.append(ho_eq_bh[-1] * (1 + r))

    ho_eq_volt = np.array(ho_eq_volt)
    ho_eq_bh   = np.array(ho_eq_bh)
    ho_r_volt, ho_sr_volt, ho_dd_volt = curve_stats(ho_eq_volt)
    ho_r_bh,   ho_sr_bh,   ho_dd_bh  = curve_stats(ho_eq_bh)

    print(f"\n  {'Strategy':<22}  {'Return':>8}  {'Sharpe':>7}  {'MaxDD':>8}")
    print(f"  {'Vol-targeted':<22}  {ho_r_volt:>8.2%}  {ho_sr_volt:>7.2f}  {ho_dd_volt:>8.2%}")
    print(f"  {'Buy & hold':<22}  {ho_r_bh:>8.2%}  {ho_sr_bh:>7.2f}  {ho_dd_bh:>8.2%}")
else:
    print("  No trained models available.")
    ho_acc_dir = ho_acc_vol = 0.0

# ---------------------------------------------------------------------------
# LIVE SIGNAL
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Live signal")
print("=" * 60)

live_dir  = "N/A"
live_p_d  = live_p_v = live_size = 0.0

if last_lr_dir is not None:
    live_row  = f_raw[feat_cols].dropna().iloc[-1:]
    row_s     = last_scaler.transform(live_row.values)
    live_p_d  = last_lr_dir.predict_proba(row_s)[0][1]
    live_p_v  = last_lr_vol.predict_proba(row_s)[0][1]
    cur_vol20 = vol20.loc[live_row.index[-1]]
    vol_adj   = 1.0 - (live_p_v - 0.5)
    live_size = float(np.clip(TARGET_VOL / max(cur_vol20, 1e-6) * vol_adj, 0.1, 3.0))

    live_dir = "LONG" if live_p_d > 0.55 else ("SHORT" if live_p_d < 0.45 else "FLAT")

    print(f"  Data through:      {live_row.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Direction signal:  {live_dir}  (p_up={live_p_d:.1%})")
    print(f"  Vol forecast:      {'expanding' if live_p_v > 0.5 else 'contracting'}  "
          f"(p_expand={live_p_v:.1%})")
    print(f"  Current vol20:     {cur_vol20:.4f}")
    print(f"  Vol-targeted size: {live_size:.2f}x")
    print(f"\n  Direction model: {'significant' if p_val_dir < 0.05 else 'NOT significant'} "
          f"vs base rate  p={p_val_dir:.4f}")
    print(f"  Vol model:       {'significant' if p_val_vol < 0.05 else 'NOT significant'} "
          f"vs base rate  p={p_val_vol:.4f}")

# ---------------------------------------------------------------------------
# PREDICTION LOG
# ---------------------------------------------------------------------------
log_entry = {
    "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "data_through": (live_row.index[-1] if last_lr_dir is not None else df.index[-1]).strftime("%Y-%m-%d"),
    "dir_signal":   live_dir,
    "p_up":         f"{live_p_d:.4f}",
    "p_vol_expand": f"{live_p_v:.4f}",
    "vol_size":     f"{live_size:.3f}",
}
log_file = "es_predictions_log.csv"
try:
    updated = pd.concat([pd.read_csv(log_file), pd.DataFrame([log_entry])], ignore_index=True)
except FileNotFoundError:
    updated = pd.DataFrame([log_entry])
updated.to_csv(log_file, index=False)
print(f"\n  Logged to {log_file}")
