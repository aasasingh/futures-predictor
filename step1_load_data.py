import pandas as pd

url = "https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv"

df = pd.read_csv(url)

print(df.head())
print("\nRows:", len(df), "Columns:", df.shape[1])
