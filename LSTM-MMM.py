#importing stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf.cast
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

register_matplotlib_converters()

#constants
test_index = 32
step_value = 20

rcParams['figure.figsize'] = 20, 10  # setting figure size

scaler = MinMaxScaler(feature_range=(0, 1))  # for normalizing data


df = pd.read_csv('Data Given/MMM.csv')
#df = pd.read_csv('NSE-BSE.csv')
print(df.head())

#setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%m/%d/%Y')
#df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
# plt.figure(figsize=(16, 8))
# plt.plot(df['Close'], label='Close Price history')
# plt.show()

#
#creating dataframe
data = df.sort_index(ascending=False, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values
# print(dataset)
train = dataset[test_index:, :]
valid = dataset[0:test_index, :]
# print(train)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)

x_train, y_train = [], []
for i in range(step_value, len(train)):
    x_train.append(scaled_data[i - step_value: i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print('x_train')
# print(x_train)
# print("y_train")
# print(y_train)


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[0:len(valid) + step_value].values

inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
print(inputs.shape[0])
for i in range(step_value, inputs.shape[0]):
    x_test.append(inputs[i - step_value: i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(x_test.shape[0])

closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)
print(closing_price)

# rms = np.sqrt(np.mean(np.power((valid-closing_price), 2)))
# rms

#for plotting
train = new_data[test_index:]
valid = new_data[:test_index]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.show()