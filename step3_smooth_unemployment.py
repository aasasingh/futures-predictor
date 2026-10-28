import pandas as pd
import matplotlib.pyplot as plt

# Load unemployment rate data
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
df = pd.read_csv(url)

# Clean columns
df.columns = ["date", "unemployment_rate"]
df["date"] = pd.to_datetime(df["date"])

# Create rolling averages
df["ma_3"] = df["unemployment_rate"].rolling(window=3).mean()
df["ma_12"] = df["unemployment_rate"].rolling(window=12).mean()

# Plot raw + smoothed
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["unemployment_rate"], alpha=0.4, label="Monthly")
plt.plot(df["date"], df["ma_3"], label="3-Month Avg")
plt.plot(df["date"], df["ma_12"], label="12-Month Avg")

plt.title("US Unemployment Rate (Smoothed)")
plt.xlabel("Year")
plt.ylabel("Percent")
plt.legend()
plt.grid(True)

plt.show()
