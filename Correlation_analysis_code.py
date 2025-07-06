import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define assets and parameters
tickers = ["GC=F", "CL=F", "BTC-USD", "TATAMOTORS.NS"]
start_date = "2022-05-06"
end_date = "2025-05-06"
interval = "1wk"

# Download data
data = yf.download(tickers, start=start_date, end=end_date, interval=interval)['Close']

# Rename columns
data.columns = ["Gold", "Oil", "Bitcoin", "Tata Motors"]
data.dropna(inplace=True)

# Print preview and correlation matrix
print(data.head())
print("\nPairwise Correlation Matrix:")
print(data.corr())

# --- Correlation Heatmaps ---
plt.figure(figsize=(20, 4))

plt.subplot(1, 4, 1)
sns.heatmap(data[["Gold", "Oil"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Gold vs Oil")

plt.subplot(1, 4, 2)
sns.heatmap(data[["Gold", "Bitcoin"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Gold vs Bitcoin")

plt.subplot(1, 4, 3)
sns.heatmap(data[["Oil", "Bitcoin"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Oil vs Bitcoin")

plt.subplot(1, 4, 4)
sns.heatmap(data[["Tata Motors", "Bitcoin"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Tata Motors vs Bitcoin")

plt.tight_layout()
plt.show()

# --- Scatter Plots with Trend Lines ---
plt.figure(figsize=(20, 8))

plt.subplot(2, 3, 1)
sns.regplot(x=data["Oil"], y=data["Gold"], scatter_kws={"s": 30}, line_kws={"color": "red"})
plt.title("Gold vs Oil")

plt.subplot(2, 3, 2)
sns.regplot(x=data["Bitcoin"], y=data["Gold"], scatter_kws={"s": 30}, line_kws={"color": "red"})
plt.title("Gold vs Bitcoin")

plt.subplot(2, 3, 3)
sns.regplot(x=data["Bitcoin"], y=data["Oil"], scatter_kws={"s": 30}, line_kws={"color": "red"})
plt.title("Oil vs Bitcoin")

plt.subplot(2, 3, 4)
sns.regplot(x=data["Bitcoin"], y=data["Tata Motors"], scatter_kws={"s": 30}, line_kws={"color": "red"})
plt.title("Tata Motors vs Bitcoin")

plt.subplot(2, 3, 5)
sns.regplot(x=data["Oil"], y=data["Tata Motors"], scatter_kws={"s": 30}, line_kws={"color": "red"})
plt.title("Tata Motors vs Oil")

plt.subplot(2, 3, 6)
sns.regplot(x=data["Gold"], y=data["Tata Motors"], scatter_kws={"s": 30}, line_kws={"color": "red"})
plt.title("Tata Motors vs Gold")

plt.tight_layout()
plt.grid(True)
plt.show()

# --- Final Combined Correlation Matrix ---
plt.figure(figsize=(7, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Combined Correlation Matrix: Gold, Oil, Bitcoin, Tata Motors")
plt.show()

# --- Final Info ---
print(f"\nNumber of weekly data points used: {len(data)}")