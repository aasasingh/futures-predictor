import pandas as pd

# Load data
un = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE")
un.columns = ["date", "unemployment"]
un["date"] = pd.to_datetime(un["date"])

yc = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M")
yc.columns = ["date", "yield_spread"]
yc["date"] = pd.to_datetime(yc["date"])
# T10Y3M is daily; UNRATE/USREC are dated the 1st of each month. Average the
# spread by month so every month lines up (an exact-date merge dropped any
# month whose 1st was a weekend or holiday).
yc = yc.set_index("date").resample("MS").mean().reset_index()

rec = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC")
rec.columns = ["date", "recession"]
rec["date"] = pd.to_datetime(rec["date"])

df = un.merge(yc, on="date").merge(rec, on="date")

# Signals
df["un_ma_3"] = df["unemployment"].rolling(3).mean()
df["un_ma_12"] = df["unemployment"].rolling(12).mean()
df["un_signal"] = df["un_ma_3"] > df["un_ma_12"]
df["yc_signal"] = df["yield_spread"] < 0

df["risk_score"] = df["yc_signal"].astype(int) * 60 + df["un_signal"].astype(int) * 40
df = df.dropna().reset_index(drop=True)

df["high_risk"] = df["risk_score"] >= 60

# Hit rate: recession occurs within H months after a high-risk month
H = 18
high_months = 0
hits = 0
leads = []

# Only signals outside a recession count; one fired mid-recession isn't a
# prediction, and it would always be a "hit" with a 0-month lead
for i in range(len(df)):
    if df["high_risk"].iloc[i] and df["recession"].iloc[i] == 0:
        high_months += 1
        future = df.iloc[i+1:i+H+1]
        if future["recession"].any():
            hits += 1
            lead = future["recession"].idxmax() - df.index[i]
            leads.append(lead)

hit_rate = hits / high_months if high_months else 0
avg_lead = sum(leads) / len(leads) if leads else None

print("High-risk months:", high_months)
print("Hits (recession within 18mo):", hits)
print("Hit rate:", round(hit_rate * 100, 1), "%")
print("Avg lead (months):", round(avg_lead, 1) if avg_lead is not None else "N/A")
