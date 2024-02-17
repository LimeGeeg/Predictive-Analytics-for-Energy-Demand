import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the original data
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Load the our saved model
with open('trained_sarima_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Generate future dates for prediction
future_dates = pd.date_range(start='2019-06-01', periods=5, freq='MS')  # Assuming 5 months of future predictions

# Make predictions for the future dates
forecast = loaded_model.forecast(steps=5)                               # Predicting 5 steps ahead

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Value': forecast})
predictions_df.set_index('Date', inplace=True)

# Plot the original data and predicted values
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Value'], label='Original Data')
plt.plot(predictions_df.index, predictions_df['Forecasted_Value'], label='Forecasted Values', color='red', linestyle='--')
plt.title('Original Data vs Forecasted Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
