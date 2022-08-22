# import packages
import warnings
import gc
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np

# to plot within notebook
import matplotlib.pyplot as plt

# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
pd.options.mode.chained_assignment = None
# for normalizing data
from sklearn.preprocessing import MinMaxScaler
import sys

plt.switch_backend('svg')
scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv(r'dataSet.csv')
#print(df)
# setting index as date values
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

# sorting
data = df.sort_index(ascending=True, axis=0)

# creating a separate dataset
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

#print(new_data)
# print('-'*100)
# print(data['Date'])
# print('-'*100)
new_data['Date'] = pd.to_datetime(new_data['Date'])
for i in range(0, len(data)):
    new_data['Close'][i] = data['Close'].copy()[i]
    new_data['Date'][i] = data['Date'].copy()[i]
    #new_data['Date'][i] = pd.to_numeric(pd.to_datetime(data['Date'].copy()[i]))

    # print(data['Date'][i])
    # print("-------" * 10)
    # print(new_data['Date'][i])
    # print("********" * 10)


    # print(new_data)
    # print("xxxxxxxx" * 10)


print(new_data)
# print(new_data['Date'])

# create features
from fastai.tabular.all import *

add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp

new_data['mon_fri'] = 0
for i in range(0, len(new_data)):
    if new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4:
        new_data['mon_fri', i] = 1
    else:
        new_data['mon_fri', i] = 0


# split into train and validation
train = new_data[:987].copy()
valid = new_data[987:].copy()

x_train = train.drop('Close', axis=1)
y_train = train['Close'].copy()
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close'].copy()

# implement linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

# make predictions and find the rmse
preds = model.predict(x_valid)

rmse = str(math.sqrt(mean_squared_error(preds, y_valid)))
rmae = str(math.sqrt(mean_absolute_error(preds, y_valid)))
f = open("LinearRegression_stats.txt", "w")
mean_squared_error_val1 = "Mean Squared Error: " + str(mean_squared_error(preds, y_valid))
RMSE = "RMSE: {}".format(rmse)
mean_absolute_error_val1 = "Mean Absolute Error: " + str(mean_absolute_error(preds, y_valid))
RMAE = "RMAE: {}".format(rmae)
R2 = "R^2: " + str(r2_score(y_valid, preds))
f.write(mean_squared_error_val1 + '\n')
f.write(RMSE + '\n')
f.write(mean_absolute_error_val1 + '\n')
f.write(RMAE + '\n')
f.write(R2 + '\n')
f.close()

valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.title("Linear Regression")
plt.plot(train['Close'], label = 'Train')
plt.plot(valid[['Close', 'Predictions']], label = "Test si Preturi prezise")
plt.legend()
plt.savefig('app/static/LinearRegression_plot.png')
#plt.show()

params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()  # alg knn regresie
model2 = GridSearchCV(knn, params, cv=5)  # cautam cel mai bun k

model2.fit(x_train, y_train)
preds2 = model2.predict(x_valid)


f = open("KNN_stats.txt", "w")
mean_squared_error_val2 = "Mean Squared Error: " + str(mean_squared_error(preds2, y_valid))
RMSE = "RMSE: {}".format(math.sqrt(mean_squared_error(preds2, y_valid)))
mean_absolute_error_val2 = "Mean Absolute Error: " + str(mean_absolute_error(preds2, y_valid))
RMAE = "RMAE: {}".format(math.sqrt(mean_absolute_error(preds2, y_valid)))
R2 = "R^2: " + str(r2_score(y_valid, preds2))
f.write(mean_squared_error_val2 + '\n')
f.write(RMSE + '\n')
f.write(mean_absolute_error_val2 + '\n')
f.write(RMAE + '\n')
f.write(R2 + '\n')
f.close()

valid['Predictions'] = 0
valid['Predictions'] = preds2

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.clf()
plt.title("KNN")
plt.plot(train['Close'], label = "Train")
plt.plot(valid[['Close', 'Predictions']], label = "Test si Preturi prezise")
plt.legend()
plt.savefig('app/static/KNN_Reg_plot.png')
#plt.show()

# del new_data
# gc.collect()
# del df
# gc.collect()
# del data
# gc.collect()
#
