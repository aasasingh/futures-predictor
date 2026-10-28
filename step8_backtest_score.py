import pandas as pd

# Load unemployment
un_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
un = pd.read_csv(un_url)
un.columns = ["date", "unemployment"]
un["date"] = pd.to_datetime(un["date"])

# Load yield curve
yc_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M"
yc = pd.read_csv(yc_url)
yc.columns = ["date", "yield_spread"]
yc["date"] = pd.to_datetime(yc["date"])
# T10Y3M is daily; UNRATE/USREC are dated the 1st of each month. Average the
# spread by month so every month lines up (an exact-date merge dropped any
# month whose 1st was a weekend or holiday).
yc = yc.set_index("date").resample("MS").mean().reset_index()

# Load recession labels
rec_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
rec = pd.read_csv(rec_url)
rec.columns = ["date", "recession"]
rec["date"] = pd.to_datetime(rec["date"])

# Merge all datasets
df = un.merge(yc, on="date").merge(rec, on="date")

# Signals
df["un_ma_3"] = df["unemployment"].rolling(3).mean()
df["un_ma_12"] = df["unemployment"].rolling(12).mean()
df["un_signal"] = df["un_ma_3"] > df["un_ma_12"]
df["yc_signal"] = df["yield_spread"] < 0

# Composite risk score
df["risk_score"] = (
    df["yc_signal"].astype(int) * 60 +
    df["un_signal"].astype(int) * 40
)

df = df.dropna().reset_index(drop=True)

# High-risk periods
df["high_risk"] = df["risk_score"] >= 60

# Overlap with recessions
high_risk_count = df["high_risk"].sum()
overlap_count = df[(df["high_risk"]) & (df["recession"] == 1)].shape[0]

overlap_rate = overlap_count / high_risk_count if high_risk_count > 0 else 0

print("High-risk periods:", high_risk_count)
print("Overlap with recessions:", round(overlap_rate * 100, 1), "%")

# Lead time calculation (months)
leads = []

# Only signals outside a recession count; one fired mid-recession isn't a lead
for i in range(len(df)):
    if df["high_risk"].iloc[i] and df["recession"].iloc[i] == 0:
        future = df.iloc[i+1:i+19]
        if future["recession"].any():
            lead = future["recession"].idxmax() - df.index[i]
            leads.append(lead)

if leads:
    print("Avg lead time (months):", round(sum(leads) / len(leads), 1))
else:
    print("No lead times found")
