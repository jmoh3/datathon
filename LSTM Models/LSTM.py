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
from sklearn.metrics import r2_score

register_matplotlib_converters()

# fix random seed for reproducibility
np.random.seed(42)

#constants
test_index = 32
step_value = 60

rcParams['figure.figsize'] = 20, 10  # setting figure size
scaler = MinMaxScaler(feature_range=(0, 1))  # for normalizing data

df = pd.read_csv('Data Given/MMM.csv')

#setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%m/%d/%y')
df.index = df['Date']


#creating dataframe
data = df.sort_index(ascending=False, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
actual_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, test_index):
    actual_data['Date'][i] = data['Date'][i]
    actual_data['Close'][i] = data['Close'][i]

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
actual_data.index = actual_data.Date
actual_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values
datasetactual = actual_data.values

# separate train and test
train = dataset[test_index:, :]
test = dataset[0:test_index, :]
actual = datasetactual[0:test_index, :]

# calculate average close and input that into test
average_close = train.mean()
for i in range(0, test_index):
    new_data['Close'][i] = average_close


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)



# divide train into x_train and y_train
x_train, y_train = [], []
for i in range(step_value, len(train)):
    x_train.append(scaled_data[i - step_value: i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# fit LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# scale test inputs and convert to x_test
inputs = new_data[0:len(test) + step_value].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(step_value, inputs.shape[0]):
    x_test.append(inputs[i - step_value: i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# use LSTM model to predict closing price
closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)
print(closing_price)

# for plotting
train = new_data[test_index:]
test = new_data[:test_index]
test['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(test[['Predictions']])
plt.plot(actual_data['Close'])
plt.show()


# Calculate errors
errors = abs(closing_price - actual_data)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / actual)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

r2 = r2_score(closing_price, actual_data)
print('R^2: ', round(r2, 2), '%.')

