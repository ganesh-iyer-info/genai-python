import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv")

# Filter for USA
usa = df[df['Country'] == 'US']

# Plot cases over time
plt.figure(figsize=(10,6))
sns.lineplot(data=usa, x='Date', y='Confirmed', label='Confirmed')
sns.lineplot(data=usa, x='Date', y='Deaths', label='Deaths')
plt.title("COVID-19 Cases in USA")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()