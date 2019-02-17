import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math


# Read in data and display first 5 rows
data = pd.read_csv('Data Given/SYF-Aggregated.csv')

#adding date columns
date = data['Date'].str.split('/', expand=True)
data['Month'] = date[0]
data['Day'] = date[1]
data['Year'] = date[2]


# Create numpy array of data without Close
data = data[np.isfinite(data['AvgClose60Days'])]
labels = np.array(data['Close'])  # Labels are the values we want to predict
dates = np.array(data['Date'])
data = data.drop('Close', axis=1)
data = data.drop('Date', axis=1)
data = data.drop('Discover', axis=1)
data = data.drop('OMF', axis=1)
data = data.drop('DOW', axis=1)
factors_list = list(data.columns)
data = np.array(data)
print(data)
# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels, train_date, test_date = train_test_split(data, labels, dates, test_size=0.030, shuffle=False)


# Get baseline prediction
average_close = train_labels.mean()
baseline_errors = abs(average_close - test_labels)
baseline_errors_squared = baseline_errors**2
average_baseline_error = round(np.mean(baseline_errors), 2)
print('Average baseline error: ', average_baseline_error)
squared_baseline_error = round(np.mean(baseline_errors_squared), 2)
print('Mean Squared baseline error: ', squared_baseline_error)


# Instantiate and train model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_data, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_data)

# Calculate errors
errors = abs(predictions - test_labels)
errorSquared = errors**2
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Mean Squared Error:', round(np.mean(errorSquared), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



r2 = r2_score(predictions, test_labels)
print('R^2: ', round(r2, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(factors_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


training = pd.DataFrame({'Date': train_date, 'Close': train_labels})
training.index = training['Date']
actual = pd.DataFrame({'Date': test_date, 'Close': test_labels})
actual.index = actual['Date']
predicted = pd.DataFrame({'Date': test_date, 'Close': predictions})
predicted.index = predicted['Date']

print()
print("THIS IS THE PREDICTED DATA")
print(predicted)
#plot
plt.figure(figsize=(16, 8))
plt.plot(training['Close'])
plt.plot(actual['Close'])
plt.plot(predicted['Close'])
plt.show()