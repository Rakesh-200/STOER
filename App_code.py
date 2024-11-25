import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from datetime import datetime
import pytz

# Load your dataset (make sure your file path is correct)
data = pd.read_csv('smart_traffic_data.csv')

# Show the first few rows of the dataset
print(data.head())

# Data Preprocessing

# Handle missing values
missing_data = data.isnull().sum()
print(f"Missing data:\n{missing_data}")

# Fill missing values with the mean (or you can choose a different method)
data.fillna(data.mean(), inplace=True)

# Convert Timestamp to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Extract additional time-based features
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month
data['Year'] = data['Timestamp'].dt.year

# Feature Engineering (optional)
data['Public_Transport_Vehicles_Ratio'] = data['Number of public transport users per hour'] / (data['Traffic flow'] + data['Number of public transport users per hour'])

# Data Exploration: Visualize traffic flow and public transport usage over time
plt.figure(figsize=(12,6))
sns.lineplot(data=data, x='Timestamp', y='Traffic flow', label='Traffic Flow')
sns.lineplot(data=data, x='Timestamp', y='Number of public transport users per hour', label='Public Transport Usage')
plt.title('Traffic Flow and Public Transport Usage Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Count')
plt.legend()
plt.show()

# Correlation analysis
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Prepare Data for Traffic Flow Prediction
X = data[['Number of public transport users per hour', 'Weather Conditions', 'Day of the week', 'Temperature', 'Humidity', 'Road Incidents']]
y = data['Traffic flow']

# Convert categorical variables (e.g., Weather Conditions, Day of the week) into numerical values
X = pd.get_dummies(X, drop_first=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest model (you can use other models like Linear Regression, Decision Tree, etc.)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Traffic Optimization Function (simple dynamic signal control)
def optimize_traffic_light(predicted_traffic_flow):
    if predicted_traffic_flow > 1000:
        return "Red Light - High Traffic, prioritize public transport"
    elif predicted_traffic_flow > 500:
        return "Green Light - Moderate Traffic"
    else:
        return "Green Light - Low Traffic"

# Emission Reduction Strategy Function
def recommend_emission_reduction(public_transport_usage, bike_sharing_usage, road_incidents):
    if public_transport_usage > 1000:
        return "Encourage more public transport usage."
    elif bike_sharing_usage > 500:
        return "Promote bike-sharing programs."
    elif road_incidents > 50:
        return "Improve road safety and reduce incidents."
    else:
        return "Emission is low, no immediate actions needed."

# Example prediction and optimization
predicted_traffic_flow = model.predict(scaler.transform(pd.DataFrame([[100, 1, 1, 25, 80, 0]])))[0]  # Example input for prediction
print("Optimized Traffic Light: ", optimize_traffic_light(predicted_traffic_flow))

# Example emission reduction recommendation
print("Emission Reduction Strategy: ", recommend_emission_reduction(1200, 200, 30))

# Visualize Traffic Flow and Emission Levels
def plot_traffic_and_emission(data):
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Plot traffic flow
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Traffic Flow', color='tab:blue')
    ax1.plot(data['Timestamp'], data['Traffic flow'], color='tab:blue', label='Traffic Flow')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create another y-axis to plot emission levels
    ax2 = ax1.twinx()
    ax2.set_ylabel('Emission Level (estimated)', color='tab:green')
    ax2.plot(data['Timestamp'], data['Traffic flow'] * 0.1, color='tab:green', label='Emission Level')  # Simple emission model
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title('Traffic Flow and Emission Levels Over Time')
    plt.show()

# Plot traffic and emission
plot_traffic_and_emission(data)

