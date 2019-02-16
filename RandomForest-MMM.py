import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Read in data and display first 5 rows
data = pd.read_csv('Data Given/MMM.csv')


#setting index as date
data['Year'] = data['Date'].apply(lambda x: int(str(x)[-4:]))
# data['Date'] = pd.to_datetime(data.Date, format='%m/%d/%Y')
# data.index = data['Date']


# One-hot encode the data using pandas get_dummies
features = pd.get_dummies(data)

# Create numpy array of data without Close
labels = np.array(data['Close'])  # Labels are the values we want to predict
data = data.drop('Close', axis=1)
data = data.drop('Date', axis=1)
factors_list = list(data.columns)
data = np.array(data)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=42)


# Get baseline prediction
average_close = labels.mean()
baseline_errors = abs(average_close - test_labels)
average_baseline_error = round(np.mean(baseline_errors), 2)
print('Average baseline error: ', average_baseline_error)


# Instantiate and train model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_data, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_data)
# Calculate errors
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(factors_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
