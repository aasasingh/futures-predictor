import pandas as pd
import numpy as np
import yfinance as yf
import os, sys
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ================================================================
#  ES v6 — PAPER TRADE TRACKER
#
#  Reads es_v6_signals.csv (written by the predictor)
#  Pulls actual ES prices for each signal date
#  Scores: did the signal match what actually happened?
#  Outputs a live scorecard + equity curve
# ================================================================

LOG_FILE = "es_v6_signals.csv"
RESULTS_FILE = "es_v6_results.csv"
MES_MULTIPLIER = 5  # $5 per point per MES contract

print("=" * 60)
print("  ES v6 — PAPER TRADE SCORECARD")
print("=" * 60)

# ── Load signal log ─────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    print(f"\n  ⚠ No signal log found at {LOG_FILE}")
    print(f"  Run es_predictor_v6.py first to generate signals.")
    sys.exit()

signals = pd.read_csv(LOG_FILE)
print(f"\n  Loaded {len(signals)} signals from {LOG_FILE}")

if len(signals) == 0:
    print("  No signals to score yet.")
    sys.exit()

# The scheduler also runs on weekends/holidays, which re-logs the last trading
# day. Keep one signal per data_date so a trade isn't scored twice.
signals = signals.drop_duplicates(subset="data_date", keep="last").reset_index(drop=True)

# ── Pull actual ES data ─────────────────────────────────────
print(f"  Pulling actual ES prices...")
es = yf.Ticker("ES=F").history(period="1y", interval="1d")

if es.index.tz is not None:
    es.index = es.index.tz_localize(None)
es.index = es.index.normalize()
# Drop today's bar: the session is still trading, so its close isn't final
today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
es = es[es.index < today]

print(f"  ES data: {len(es)} rows ({es.index[0].strftime('%Y-%m-%d')} → {es.index[-1].strftime('%Y-%m-%d')})")

# ── Score each signal ────────────────────────────────────────
results = []

for idx, row in signals.iterrows():
    data_date = pd.to_datetime(row["data_date"])
    signal    = row["signal"]
    size      = row.get("size_mes", 0)
    dir_prob  = float(row.get("dir_prob", 0.5))
    vol_prob  = float(row.get("vol_prob", 0.5))
    regime    = row.get("regime", "")
    favorable = row.get("regime_favorable", False)

    # Find the NEXT trading day after data_date (that's when you'd trade)
    future_dates = es.index[es.index > data_date]
    if len(future_dates) == 0:
        # Can't score yet — market hasn't happened
        results.append({
            "data_date": data_date.strftime("%Y-%m-%d"),
            "signal": signal,
            "size": size,
            "status": "PENDING",
            "trade_date": "",
            "es_open": "",
            "es_close": "",
            "direction_correct": "",
            "pnl_points": "",
            "pnl_dollars": "",
        })
        continue

    trade_date = future_dates[0]
    trade_row  = es.loc[trade_date]
    es_open    = trade_row["Open"]
    es_close   = trade_row["Close"]
    es_change  = es_close - es_open
    es_pct     = es_change / es_open * 100

    if signal == "FLAT" or signal == "N/A":
        results.append({
            "data_date": data_date.strftime("%Y-%m-%d"),
            "signal": signal,
            "size": 0,
            "status": "NO TRADE",
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "es_open": f"{es_open:.2f}",
            "es_close": f"{es_close:.2f}",
            "es_change_pct": f"{es_pct:.2f}%",
            "direction_correct": "",
            "pnl_points": 0,
            "pnl_dollars": 0,
        })
        continue

    # Score the trade
    if signal == "BUY":
        pnl_points = es_change
        correct = es_change > 0
    elif signal == "SELL":
        pnl_points = -es_change
        correct = es_change < 0
    else:
        pnl_points = 0
        correct = None

    pnl_dollars = pnl_points * MES_MULTIPLIER * int(size) if size else 0

    results.append({
        "data_date": data_date.strftime("%Y-%m-%d"),
        "signal": signal,
        "size": size,
        "status": "SCORED",
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "es_open": f"{es_open:.2f}",
        "es_close": f"{es_close:.2f}",
        "es_change_pct": f"{es_pct:.2f}%",
        "direction_correct": "✓" if correct else "✗",
        "pnl_points": f"{pnl_points:.2f}",
        "pnl_dollars": f"{pnl_dollars:.2f}",
    })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False)

# ── Print scorecard ──────────────────────────────────────────
print(f"\n{'='*60}")
print("  TRADE LOG")
print(f"{'='*60}")

scored   = results_df[results_df["status"] == "SCORED"]
pending  = results_df[results_df["status"] == "PENDING"]
no_trade = results_df[results_df["status"] == "NO TRADE"]

if len(scored) > 0:
    print(f"\n  {'Date':<12} {'Signal':<6} {'Size':<5} {'ES Open':>10} {'ES Close':>10} {'Chg':>8} {'Hit':>4} {'P&L':>10}")
    print(f"  {'─'*70}")
    for _, r in scored.iterrows():
        print(f"  {r['trade_date']:<12} {r['signal']:<6} {r['size']:<5} "
              f"{r['es_open']:>10} {r['es_close']:>10} {r['es_change_pct']:>8} "
              f"{r['direction_correct']:>4} ${r['pnl_dollars']:>9}")

if len(no_trade) > 0:
    print(f"\n  Flat days (no trade): {len(no_trade)}")

if len(pending) > 0:
    print(f"  Pending (market hasn't opened): {len(pending)}")

# ── Summary stats ────────────────────────────────────────────
print(f"\n{'='*60}")
print("  PERFORMANCE SUMMARY")
print(f"{'='*60}")

if len(scored) > 0:
    wins     = scored[scored["direction_correct"] == "✓"]
    losses   = scored[scored["direction_correct"] == "✗"]
    n_trades = len(scored)
    n_wins   = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0

    pnl_list = scored["pnl_dollars"].astype(float)
    total_pnl = pnl_list.sum()
    avg_win   = pnl_list[pnl_list > 0].mean() if len(pnl_list[pnl_list > 0]) > 0 else 0
    avg_loss  = abs(pnl_list[pnl_list < 0].mean()) if len(pnl_list[pnl_list < 0]) > 0 else 0
    pf        = avg_win / avg_loss if avg_loss > 0 else 0

    # Equity curve
    equity = [0]
    for pnl in pnl_list:
        equity.append(equity[-1] + pnl)
    max_eq   = max(equity)
    max_dd   = min(np.array(equity) - np.maximum.accumulate(equity))

    print(f"""
  Total signals:   {len(signals)}
  Trades taken:    {n_trades}  ({len(no_trade)} flat days skipped)
  Wins:            {n_wins}
  Losses:          {n_losses}
  Win rate:        {win_rate:.1f}%

  Total P&L:       ${total_pnl:>10,.2f}
  Avg win:         ${avg_win:>10,.2f}
  Avg loss:        ${avg_loss:>10,.2f}
  Profit factor:   {pf:.2f}
  Max drawdown:    ${max_dd:>10,.2f}

  ASSESSMENT:""")

    if n_trades < 10:
        print(f"    ⏳ Too few trades to assess ({n_trades}/30 minimum). Keep logging.")
    elif n_trades < 30:
        print(f"    ⏳ Building sample ({n_trades}/30). Preliminary: {'PROMISING' if win_rate > 53 else 'NEEDS MONITORING'}")
    else:
        if win_rate > 55 and pf > 1.2:
            print(f"    ✓ EDGE CONFIRMED — consider live trading with 1 MES")
        elif win_rate > 50 and pf > 1.0:
            print(f"    ~ MARGINAL — continue paper trading for 30 more signals")
        else:
            print(f"    ✗ NO EDGE — direction model failed live. Pivot to vol-only strategy.")

else:
    print(f"\n  No scored trades yet. Run the predictor for a few days,")
    print(f"  then run this tracker to see results.")

print(f"\n  Results saved to {RESULTS_FILE}")
print(f"\n  TIP: Run this tracker anytime to update scores:")
print(f"       py es_v6_tracker.py")
