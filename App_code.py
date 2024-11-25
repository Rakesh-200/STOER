import streamlit as st
import pandas as pd
import vaex
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set the title of the app
st.title('Smart Traffic Optimization and Emission Reduction System')

# Upload the dataset
st.sidebar.header('Upload Your Dataset')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Cache the data processing to speed up file loading using vaex
@st.cache_data
def load_data_vaex(file):
    # Use vaex to read large csv
    df = vaex.open(file)
    # Drop rows with null values efficiently using vaex
    df = df.dropna()
    return df

if uploaded_file is not None:
    # Load the dataset and preprocess
    data = load_data_vaex(uploaded_file)
    
    st.write("Dataset Loaded Successfully!")
    st.write(data.head())  # Display the first few rows of the dataset

    # Data Preprocessing
    st.header("Data Preprocessing")

    # Convert 'Timestamp' to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Extract time features
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    data['Month'] = data['Timestamp'].dt.month
    data['Year'] = data['Timestamp'].dt.year

    # Feature Engineering (optional)
    data['Public_Transport_Vehicles_Ratio'] = data['Number of public transport users per hour'] / \
        (data['Traffic flow'] + data['Number of public transport users per hour'])

    # Show the processed data
    st.write(data.head())

    # Data Visualization
    st.header("Data Visualization")

    # Traffic Flow vs. Public Transport Usage over Time
    if st.checkbox('Show Traffic Flow and Public Transport Usage'):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data.to_pandas(), x='Timestamp', y='Traffic flow', label='Traffic Flow', ax=ax)
        sns.lineplot(data=data.to_pandas(), x='Timestamp', y='Number of public transport users per hour', label='Public Transport Usage', ax=ax)
        plt.title('Traffic Flow and Public Transport Usage Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Count')
        plt.legend()
        st.pyplot(fig)

    # Correlation Analysis
    if st.checkbox('Show Correlation Matrix'):
        corr_matrix = data.to_pandas().corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        plt.title('Correlation Matrix')
        st.pyplot(fig)

    # Prepare Data for Model Training
    X = data[['Number of public transport users per hour', 'Weather Conditions', 'Day of the week', 'Temperature', 'Humidity', 'Road Incidents']]
    y = data['Traffic flow']

    # Convert categorical variables to numeric
    X = pd.get_dummies(X.to_pandas(), drop_first=True)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.to_pandas(), test_size=0.2, random_state=42)

    # Train Random Forest Model
    st.header("Train the Model")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")

    # User Input for Traffic Flow Prediction
    st.header("Predict Traffic Flow")
    public_transport_users = st.number_input("Number of Public Transport Users per Hour", min_value=0)
    weather_condition = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
    day_of_week = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    temperature = st.slider("Temperature (°C)", -20, 40)
    humidity = st.slider("Humidity (%)", 0, 100)
    road_incidents = st.number_input("Number of Road Incidents", min_value=0)

    # Process user inputs
    input_data = pd.DataFrame([[public_transport_users, weather_condition, day_of_week, temperature, humidity, road_incidents]])
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data_scaled = scaler.transform(input_data)

    # Make traffic flow prediction
    if st.button("Predict Traffic Flow"):
        predicted_traffic_flow = model.predict(input_data_scaled)[0]
        st.write(f"Predicted Traffic Flow: {predicted_traffic_flow:.2f} vehicles per hour")

    # Traffic Optimization Function
    def optimize_traffic_light(predicted_traffic_flow):
        if predicted_traffic_flow > 1000:
            return "Red Light - High Traffic, prioritize public transport"
        elif predicted_traffic_flow > 500:
            return "Green Light - Moderate Traffic"
        else:
            return "Green Light - Low Traffic"

    # Display Traffic Light Optimization Suggestion
    st.header("Traffic Light Optimization")
    if st.button("Optimize Traffic Light"):
        optimized_traffic = optimize_traffic_light(predicted_traffic_flow)
        st.write(f"Optimized Traffic Light: {optimized_traffic}")

    # Emission Reduction Strategy
    def recommend_emission_reduction(public_transport_usage, bike_sharing_usage, road_incidents):
        if public_transport_usage > 1000:
            return "Encourage more public transport usage."
        elif bike_sharing_usage > 500:
            return "Promote bike-sharing programs."
        elif road_incidents > 50:
            return "Improve road safety and reduce incidents."
        else:
            return "Emission is low, no immediate actions needed."

    st.header("Emission Reduction Strategy")
    bike_sharing_usage = st.number_input("Number of Bikes Used per Hour", min_value=0)
    emission_strategy = recommend_emission_reduction(public_transport_users, bike_sharing_usage, road_incidents)
    st.write(f"Emission Reduction Strategy: {emission_strategy}")

else:
    st.warning("Please upload a CSV file to proceed.")
