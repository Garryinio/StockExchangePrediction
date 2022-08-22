import pandas as pd  # To read your dataset
import matplotlib.pyplot as plt  # pentru plotare
import plotly.express as px

#df = pd.read_csv(r'dataSet.csv')

# fig = px.line(df, x = 'Date', y = 'Close', title='Apple Share Prices over time (2014)')
# fig.show()
from matplotlib import rcParams

rcParams['figure.figsize'] = 20, 10
columns = ["Date", "Close"]
df = pd.read_csv(r'TSLA.csv', usecols=columns)


plt.plot(df.Date, df.Close)
plt.show()