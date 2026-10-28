import pandas as pd
import matplotlib.pyplot as plt

# Load yield curve spread (10Y - 3M)
spread_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M"
spread = pd.read_csv(spread_url)
spread.columns = ["date", "yield_spread"]
spread["date"] = pd.to_datetime(spread["date"])

# Load recession indicator
rec_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
rec = pd.read_csv(rec_url)
rec.columns = ["date", "recession"]
rec["date"] = pd.to_datetime(rec["date"])

# Merge: carry each month's recession flag forward onto the daily spread
# (an exact-date merge kept only days that fell on the 1st of a month)
df = pd.merge_asof(spread.dropna().sort_values("date"), rec.sort_values("date"), on="date")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["yield_spread"], label="10Y - 3M Spread")
plt.axhline(0, color="black", linestyle="--")  # inversion threshold

# Shade recessions (monthly, from the first spread date on)
rec = rec[rec["date"] >= df["date"].iloc[0]].reset_index(drop=True)
for i in range(len(rec) - 1):
    if rec["recession"].iloc[i] == 1:
        plt.axvspan(rec["date"].iloc[i], rec["date"].iloc[i + 1], color="gray", alpha=0.3)

plt.title("Yield Curve Inversion with NBER Recessions")
plt.xlabel("Year")
plt.ylabel("Percent")
plt.legend()
plt.grid(True)

plt.show()
plt.close()
