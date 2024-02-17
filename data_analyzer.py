import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

energy_data = pd.read_csv('EER2023-Monthly-Data-1-1.csv')   # Load the data

# print(energy_data.head())                                 # Display the first few rows of the DataFrame to get an overview of the data
# print(energy_data.describe())                             # Basic (summary) statistics of the numerical columns
# print(energy_data.isnull().sum())                         # Check for missing values        
                                                            
print(energy_data.describe())

# Data visualization
plt.figure(figsize=(12, 6))
plt.plot(energy_data['Date'], energy_data['Value'])
plt.title('Energy Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.show()

# Correlation analysis
correlation_matrix = energy_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

energy_data['Energy_Consumption_Lag1'] = energy_data['Energy_Consumption'].shift(1)
energy_data['Energy_Consumption_Lag7'] = energy_data['Energy_Consumption'].shift(7)

print(energy_data.head())                                   # Return modified DataFrame