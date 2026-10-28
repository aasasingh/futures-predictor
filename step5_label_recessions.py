import pandas as pd
import matplotlib.pyplot as plt

# Load unemployment rate
unrate_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
unrate = pd.read_csv(unrate_url)
unrate.columns = ["date", "unemployment_rate"]
unrate["date"] = pd.to_datetime(unrate["date"])

# Load recession indicator (NBER)
recession_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
recession = pd.read_csv(recession_url)
recession.columns = ["date", "recession"]
recession["date"] = pd.to_datetime(recession["date"])

# Merge datasets
df = pd.merge(unrate, recession, on="date", how="inner")

# Rolling averages
df["ma_3"] = df["unemployment_rate"].rolling(3).mean()
df["ma_12"] = df["unemployment_rate"].rolling(12).mean()

# Signal
df["signal"] = df["ma_3"] > df["ma_12"]

# Plot unemployment
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["unemployment_rate"], alpha=0.4, label="Unemployment")
plt.plot(df["date"], df["ma_3"], label="3-Month Avg")
plt.plot(df["date"], df["ma_12"], label="12-Month Avg")

# Shade recessions
for i in range(len(df) - 1):
    if df["recession"].iloc[i] == 1:
        plt.axvspan(df["date"].iloc[i], df["date"].iloc[i + 1], color="gray", alpha=0.3)

plt.title("Unemployment Rate with NBER Recessions")
plt.xlabel("Year")
plt.ylabel("Percent")
plt.legend()
plt.grid(True)

plt.show()
