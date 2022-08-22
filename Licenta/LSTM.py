import math

import numpy
import pandas as pd  # To read your dataset
import matplotlib.pyplot as plt  # pentru plotare
from sklearn.preprocessing import MinMaxScaler  # to scale the data (scale the data between min and max value)
from tensorflow.keras.models import Sequential, load_model  # sequence of layers for owner network
from tensorflow.keras.layers import LSTM, Dense, Dropout  # long short term memory network to feed in time data
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

import tensorflow as tf

df = pd.read_csv(r'dataSet.csv')  # read the data
# print(df.head())
# pentru a antrena programul, trebuie sa-i dam train cu una din coloanele din excel.
# aici luam cu coloana open
df = df['Close'].values
# ii dam reshape pentru a avea 2 dimensiuni
df = df.reshape(-1, 1)
print(df.shape + df[:7])
# luam doua dataset-uri, de train si de test, care vor prezice pretul stock-ului de maine preluand datele ultimelor 50
#  dam split la 1259 de date in doua dataset-uri                                                              de zile
dataset_train = numpy.array(df[:int(df.shape[0] * 0.8)])
dataset_test = numpy.array(df[int(df.shape[0] * 0.8) - 50:])
print(dataset_test.shape, dataset_train.shape)

# scaling the data
# min = 0 , max = 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)  # scalarea in sine ,transforma datele din excel in 0 si 1
print(dataset_train[:7])

# test data trebuie sa nu fie cunoscut de catre network
dataset_test = scaler.transform(dataset_test)
print(dataset_test[:7])


def create_my_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i - 50:i, 0])  # primele 50 de intrari, 0 - 49
        y.append(df[i, 0])  # a 50-a valoare
    x = numpy.array(x)
    y = numpy.array(y)
    return x, y


# data set-urile pentru train
x_train, y_train = create_my_dataset(dataset_train)
print(x_train[:1])
print(x_train[:1].shape)

# data set-urile pentru test
x_test, y_test = create_my_dataset(dataset_test)
print(x_test[:1])
print(x_test[:1].shape)
# reshape data
x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 1 reprezinta feature-urile
x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Urmatorul pas, sa cream network-ul si sa-i dam train
model = Sequential()  # network-ul
model.add(LSTM(units=96, return_sequences=True,
               input_shape=(x_train.shape[1], 1)))  # long short term memory, units sunt neuronii, sa dam return la
# input shape-ul sunt cele 50 de intrari
# dam add la dropout layer
model.add(Dropout(0.2))  # 20% din neuroni nu o sa fie folositi
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))  # nu dam return, doar ultimul input o sa fie adaugat
model.add(Dropout(0.2))
model.add(Dense(units=1))  # la final avem doar un output
print(model.summary())

# defining the loss and train
model.compile(loss='mean_squared_error', optimizer='adam')
# fiting the training data to the model
if (not os.path.exists(r'stock_prediction.h5')):
    model.fit(x_train, y_train, epochs=50, batch_size=32)  # epochs = parcurge training data-ul de 50 de ori, batch = 32
    # de exemple
    model.save(r'stock_prediction.h5')

model = load_model(r'stock_prediction.h5')
predictions = model.predict(x_test)

#statistici
# print("Mean Squared Error: ", mean_squared_error(predictions, y_test))
# print("RMSE: ", math.sqrt(mean_squared_error(predictions, y_test)))
# print("Mean Absolute Error: ", mean_absolute_error(predictions, y_test))
# print("RMAE: ", math.sqrt(mean_absolute_error(predictions, y_test)))
# print("R^2: ", r2_score(y_test,predictions))

rmse = str(math.sqrt(mean_squared_error(predictions, y_test)))
rmae = str(math.sqrt(mean_absolute_error(predictions, y_test)))

f = open("LSTM_stats.txt", "w")
mean_squared_error_val1 = "Mean Squared Error: " + str(mean_squared_error(predictions, y_test))
RMSE = "RMSE: {}".format(rmse)
mean_absolute_error_val1 = "Mean Absolute Error: " + str(mean_squared_error(predictions, y_test))
RMAE = "RMAE: {}".format(rmae)
R2 = "R^2: " + str(r2_score(y_test,predictions))
f.write(mean_squared_error_val1 + '\n')
f.write(RMSE + '\n')
f.write(mean_absolute_error_val1 + '\n')
f.write(RMAE + '\n')
f.write(R2 + '\n')
f.close()



# inversam scalarea
predictions = scaler.inverse_transform(predictions)
# print(predictions)

fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(df, color='red', label='Pret real')
ax.plot(range(len(y_train) + 50, len(y_train) + 50 + len(predictions)), predictions, color='blue', label='Predictii')
plt.legend()
print(range(len(y_train) + 50, len(y_train) + 50 + len(predictions)))
plt.savefig('app/static/LSTM_plot1.png')
#plt.show()

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(y_test_scaled, color='red', label='Pret real')
plt.plot(predictions, color='blue', label='Predictii')
plt.legend()
plt.savefig('app/static/LSTM_plot2.png')
plt.close()
#plt.show()
