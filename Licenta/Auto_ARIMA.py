# import packages
import math
import os

import pandas as pd
import numpy as np

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv(r'dataSet.csv')

from pmdarima.arima import auto_arima

data = df.sort_index(ascending=True, axis=0)

train = data[:987].copy()
valid = data[987:].copy()

training = train['Close']
validation = valid['Close']

model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                   trace=True,
                   error_action='ignore', suppress_warnings=True)

model.fit(training)

forecast = model.predict(n_periods=272)

if os.path.exists("Auto_ARIMA_stats.txt"):
    os.remove("Auto_ARIMA_stats.txt")

rmse = str(math.sqrt(mean_squared_error(forecast, validation)))
rmae = str(math.sqrt(mean_absolute_error(forecast, validation)))
f = open("Auto_ARIMA_stats.txt", "w")
mean_squared_error = "Mean Squared Error: " + str(mean_squared_error(forecast, validation))
RMSE = "RMSE: {}".format(rmse)
mean_absolute_error = "Mean Absolute Error: " + str(mean_absolute_error(forecast, validation))
RMAE = "RMAE: {}".format(rmae)
R2 = "R^2: " + str(r2_score(validation,forecast))
f.write(mean_squared_error + '\n')
f.write(RMSE + '\n')
f.write(mean_absolute_error + '\n')
f.write(RMAE + '\n')
f.write(R2 + '\n')
f.close()
# print("Mean Squared Error: ", mean_squared_error(forecast, validation))
# print("RMSE: ", math.sqrt(mean_squared_error(forecast, validation)))
# print("Mean Absolute Error: ", mean_absolute_error(forecast, validation))
# print("RMAE: ", math.sqrt(mean_absolute_error(forecast, validation)))
# print("R^2: ", r2_score(validation,forecast))

forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

rms = np.sqrt(np.mean(np.power((np.array(valid['Close']) - np.array(forecast['Prediction'])), 2)))



# plot
plt.clf()
plt.title("Auto_ARIMA")
plt.plot(train['Close'],label='Train')
plt.plot(valid['Close'], label='Test')
plt.plot(forecast['Prediction'],label='Preturi prezise')
plt.legend()
plt.savefig('app/static/Auto_Arima_plot.png')
# plt.close()
# plt.close()

