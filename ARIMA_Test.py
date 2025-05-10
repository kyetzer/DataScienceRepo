import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima  # For easier ARIMA order selection
from sklearn.metrics import mean_squared_error
from math import sqrt

# 1. Generate or Load Time Series Data
# For this example, let's generate an ARIMA time series.  This is helpful
# because we *know* the underlying AR and MA parameters.  In a real
# application, you'd load your data from a CSV or other source.
#
# The following code generates a series with:
#   - AR(1) component with coefficient 0.7
#   - MA(1) component with coefficient 0.3
#   - A bit of random noise
np.random.seed(10)  # for reproducibility
n_samples = 200
ar_params = [0.7]
ma_params = [0.3]
noise = np.random.normal(0, 1, n_samples)  # White noise
# Generate ARMA process
y = np.zeros(n_samples)
y[0] = noise[0]  # Initial value
for t in range(1, n_samples):
    y[t] = ar_params[0] * y[t-1] + noise[t] + ma_params[0] * noise[t-1]
# Create a date range for plotting
date_rng = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
data = pd.Series(y, index=date_rng)

# 1b.  (Regular case) Load time series data
# If you have a CSV, you'd load it like this:
# data = pd.read_csv('your_time_series_data.csv', index_col='Date', parse_dates=True)
#
#  Make sure your date column is the index, and it's a datetime type.
#  For example, if your date column is called 'Date', and your value
#  column is called 'Value', you'd do:
# data = pd.read_csv('my_time_series.csv')
# data['Date'] = pd.to_datetime(data['Date'])
# data = data.set_index('Date')
# data = data['Value'] # keep only the value column.


# 2. Visualize the Time Series
plt.figure(figsize=(10, 4))
plt.plot(data, label='Time Series Data')
plt.title('Generated Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()


# 3. Check for Stationarity
#   - Use the Augmented Dickey-Fuller (ADF) test.
#   - The null hypothesis of the ADF test is that the time series is non-stationary.
#   - If the p-value is less than a significance level (e.g., 0.05), we reject the null
#     hypothesis and conclude that the series is stationary.
def adf_test(series):
    result = adfuller(series)
    print('ADF Test:')
    print(f'ADF Statistic: {result[0]:.3f}')
    print(f'p-value: {result[1]:.3f}')
    print(f'Critical Values: {result[4]}')
    if result[1] <= 0.05:
        print("Conclusion: The series is likely stationary.")
    else:
        print("Conclusion: The series is likely non-stationary.")

adf_test(data)

# 4. Make the Time Series Stationary (if necessary)
#   - If the ADF test indicates non-stationarity, we need to difference the series.
#   - We'll use differencing until the series becomes stationary.
#   - In this example, the generated data *is* stationary, so we don't need to difference.
#   - But, I'll include the code here for completeness.  In a real-world
#     scenario, you'd uncomment and use this if needed.

# If the series is not stationary, difference it:
# d = 0
# data_diff = data.copy()
# while True:
#     result = adfuller(data_diff.dropna())
#     if result[1] <= 0.05:
#         print(f"Differencing order: {d}")
#         break
#     else:
#         d += 1
#         data_diff = data_diff.diff()
#
# # Use the differenced data for further analysis
# data_stationary = data_diff.dropna()

data_stationary = data # No differencing needed for this example

# 5. Determine the Order of ARIMA (p, d, q)
#   - Use ACF and PACF plots to get initial estimates of p and q.
#   - Use auto_arima to automatically select the best ARIMA order.

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(data_stationary, ax=plt.gca(), title='ACF')
plt.subplot(212)
plot_pacf(data_stationary, ax=plt.gca(), title='PACF')
plt.show()

# Use auto_arima to find the best ARIMA order
auto_arima_model = auto_arima(data_stationary,
                      start_p=0, start_q=0,
                      max_p=5, max_q=5,  # You can adjust these
                      m=1,             # Frequency of the time series (1 for daily)
                      d=0,          # Don't force a value for d.
                      seasonal=False,   # Set to True if you have seasonality
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(auto_arima_model.summary())
# Get the order (p, d, q) from the best model
p, d, q = auto_arima_model.order
print(f"Optimal ARIMA order: ({p}, {d}, {q})")

# 6. Split Data into Training and Testing Sets
#   - Split the data into training and testing sets.
#   - We'll train the ARIMA model on the training set and evaluate its performance on the testing set.
train_size = int(len(data_stationary) * 0.8)  # 80% for training
train_data = data_stationary[:train_size]
test_data = data_stationary[train_size:]

# 7. Train the ARIMA Model
#   - Create and fit the ARIMA model using the training data and the determined order (p, d, q).
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()

# Print the model summary (optional, but provides details about the model)
print(model_fit.summary())

# 8. Make Predictions
#   - Use the trained model to make predictions on the test set.
predictions = model_fit.forecast(steps=len(test_data))
predictions = pd.Series(predictions, index=test_data.index) #make index match test

# 9. Evaluate the Model
#   - Calculate the Root Mean Squared Error (RMSE) to evaluate the model's performance.
rmse = sqrt(mean_squared_error(test_data, predictions))
print(f'RMSE: {rmse:.3f}')

# 10. Plot Results
#    - Plot the original time series, the training data, the test data, and the predictions.
plt.figure(figsize=(12, 6))
plt.plot(data_stationary, label='Original Data', color='blue')
plt.plot(train_data, label='Training Data', color='green')
plt.plot(test_data, label='Test Data', color='red')
plt.plot(predictions, label='Predictions', color='purple')
plt.title('ARIMA Model Results')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# 11.  Forecast Future Values (Optional)
#    - If you want to forecast beyond the test set, you can use the model to
#      predict future values.
# future_steps = 30  # Forecast 30 days into the future
# forecast = model_fit.forecast(steps=future_steps)
# forecast_index = pd.date_range(start=data_stationary.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
# forecast_series = pd.Series(forecast, index=forecast_index)
#
# plt.figure(figsize=(12, 6))
# plt.plot(data_stationary, label='Original Data', color='blue')
# plt.plot(forecast_series, label='Forecast', color='orange')
# plt.title('ARIMA Forecast')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
