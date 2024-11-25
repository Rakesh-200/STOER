import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# --------- Data Preprocessing ---------

@st.cache
def load_and_process_data(uploaded_file=None):
    """
    Load and process the traffic data from the uploaded file.
    """
    if uploaded_file is not None:
        # Read the file as a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        st.write(f"Successfully loaded uploaded CSV file.")
    else:
        st.error("No file provided or uploaded.")
        return None

    # Preprocess the data (handling missing values, types, etc.)
    df = df.dropna()  # Drop missing values
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert to datetime
    df['Day of Week'] = df['Timestamp'].dt.dayofweek  # Extract day of week
    
    # Convert categorical columns to category type for optimization
    df['Event'] = df['Event'].astype('category')
    df['Weather Conditions'] = df['Weather Conditions'].astype('category')
    
    return df

# --------- Traffic Optimization ---------

def train_traffic_model(df):
    """
    Train a Random Forest model to predict traffic flow based on features.
    """
    # Features and target variable
    features = ['Number of public transport users per hour', 'Bike sharing usage', 
                'Pedestrian count', 'Temperature', 'Humidity', 'Road Incidents', 
                'Public transport delay', 'Bike availability', 'Pedestrian incidents']
    target = 'Traffic flow'
    
    # Select features and target
    X = df[features]
    y = df[target]
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def predict_traffic(model, new_data):
    """
    Predict traffic flow for new data using the trained model.
    """
    return model.predict(new_data)

# --------- Emission Reduction ---------

def recommend_emission_reduction(traffic_flow, weather_conditions, public_transport_usage):
    """
    Provide emission reduction recommendations based on traffic flow and conditions.
    """
    if traffic_flow > 5000 and weather_conditions == 'Clear':
        return "Optimize traffic lights and encourage bike usage."
    elif public_transport_usage > 1000:
        return "Increase public transport frequency."
    else:
        return "Promote carpooling and use eco-friendly vehicles."

# --------- Streamlit Interface ---------

# Streamlit Title and Description
st.title("Smart Traffic Optimization and Emission Reduction System")
st.write("""
    This system predicts traffic flow and suggests actions for emission reduction based on traffic and environmental factors.
""")

# File upload section
st.header("Upload Your Traffic Data CSV")
uploaded_file = st.file_uploader("Upload a CSV file containing traffic data", type="csv")

# If the user uploaded a file, process it
if uploaded_file is not None:
    data = load_and_process_data(uploaded_file=uploaded_file)

    # Proceed with model training and prediction only if data is available
    if data is not None:
        # Train the traffic model
        traffic_model = train_traffic_model(data)

        # User inputs for traffic flow prediction
        st.header("Predict Traffic Flow")
        public_transport_users = st.number_input('Number of Public Transport Users (per hour)', min_value=0)
        bike_sharing_usage = st.number_input('Bike Sharing Usage (per hour)', min_value=0)
        pedestrian_count = st.number_input('Pedestrian Count (per hour)', min_value=0)
        temperature = st.number_input('Temperature (°C)', min_value=-50, max_value=50)
        humidity = st.number_input('Humidity (%)', min_value=0, max_value=100)
        road_incidents = st.number_input('Road Incidents (per hour)', min_value=0)
        public_transport_delay = st.number_input('Public Transport Delay (minutes)', min_value=0)
        bike_availability = st.number_input('Bike Availability (per hour)', min_value=0)
        pedestrian_incidents = st.number_input('Pedestrian Incidents (per hour)', min_value=0)

        # Prepare the input data for prediction
        input_data = {
            'Number of public transport users per hour': [public_transport_users],
            'Bike sharing usage': [bike_sharing_usage],
            'Pedestrian count': [pedestrian_count],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Road Incidents': [road_incidents],
            'Public transport delay': [public_transport_delay],
            'Bike availability': [bike_availability],
            'Pedestrian incidents': [pedestrian_incidents]
        }

        input_df = pd.DataFrame(input_data)

        # Predict traffic flow based on user input
        if st.button('Predict Traffic Flow'):
            traffic_flow_prediction = predict_traffic(traffic_model, input_df)
            st.write(f"Predicted Traffic Flow: {traffic_flow_prediction[0]:.2f} vehicles per hour")

        # User inputs for emission reduction recommendation
        st.header("Recommend Emission Reduction Strategy")
        weather_conditions = st.selectbox('Weather Conditions', ['Clear', 'Cloudy', 'Rainy', 'Snowy'])
        public_transport_usage = st.number_input('Public Transport Usage (per hour)', min_value=0)

        # Provide emission reduction recommendation
        if st.button('Get Emission Reduction Recommendation'):
            emission_recommendation = recommend_emission_reduction(
                traffic_flow_prediction[0], weather_conditions, public_transport_usage)
            st.write(f"Recommended Strategy: {emission_recommendation}")
else:
    st.write("Please upload a CSV file to get started.")
