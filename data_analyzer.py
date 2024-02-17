import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

# from sklearn.metrics import mean_absolute_error, mean_squared_error

# print(df.head())                                              # Display the first few rows of the DataFrame to get an overview of the data
# print(df.describe())                                          # Basic (summary) statistics of the numerical columns
# print(df.isnull().sum())                                      # Check for missing values  

class EnergyDemand():

    def __init__(self, validation_size: int, data_file = pd.read_csv('data.csv')):
        self.data_file = data_file
        self.validation_size = validation_size
        self.p = 1                                              # Adjust SARIMA parameters
        self.d = 1
        self.q = 1

    def data_frame(self):

        # Data visualization
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_file['Date'], self.data_file['Value'])
        plt.title('Energy Consumption Over Time')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.show()

        # Correlation analysis
        correlation_matrix = self.data_file.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

        return self.data_file.head()

    def univariate_data(self):

        electricity_demand_data = self.data_file[self.data_file['Category'] == 'Electricity demand']       # Filter the data to include only rows related to electricity demand

        time_series_data = electricity_demand_data[['Date', 'Value']]                                      # Extracting relevant columns (Date and Value)

        time_series_data['Date'] = pd.to_datetime(time_series_data['Date'])                                # Converting 'Date' column to datetime format

        time_series_data = time_series_data.groupby('Date').sum()                                          # Aggregate values for each date (will use sum as an example)

        print(time_series_data.head())                                                                     # Display the first few rows of the time series data
        print(time_series_data.value_counts())
        return

    def model_train(self):

        # Loading the data and preprocess if necessary
        self.data_file['Date'] = pd.to_datetime(self.data_file['Date'])
        self.data_file.set_index('Date', inplace=True)

        # Split the data into training and validation sets
        train_data = self.data_file['Value'][:self.validation_size]
        valid_data = self.data_file['Value'][-self.validation_size:]
        main_data = pd.read_csv('EER2023-Monthly-Data-1-1.csv')['Value']

        # Fit the SARIMA model
        order = (self.p, self.d, self.q)                                       # Specify SARIMA order
        model = SARIMAX(train_data, order=order)
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")                              # Suppress convergence warning
                sarima_model = model.fit()
        except:
            print("Model did not converge!")

        # Make predictions
        forecast = sarima_model.predict(start=len(train_data), end=len(train_data) + len(valid_data) - 1, dynamic=False)

        # Plot original data and forecasted values
        plt.figure(figsize=(12, 6))
        plt.plot(main_data.index, main_data, label='Main Data')
        plt.plot(valid_data.index, valid_data, label='Validation Data')
        plt.plot(forecast.index, forecast, label='Forecast', color='red')
        plt.title('SARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.legend()
        plt.show()

        # Calculate MAE, MSE, RMSE
        # mae = mean_absolute_error(valid_data, forecast)
        # mse = mean_squared_error(valid_data, forecast)
        # rmse = np.sqrt(mse)

        # print("Mean Absolute Error (MAE):", mae)
        # print("Mean Squared Error (MSE):", mse)
        # print("Root Mean Squared Error (RMSE):", rmse)


model = EnergyDemand(1200)
# model.data_frame()              # Visualize and explore data
forecast = model.model_train()    # Train model

print(forecast)                   # Print or visualize the forecasted values