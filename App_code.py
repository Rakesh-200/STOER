import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
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
        
        # Strip any leading/trailing spaces in column names for accuracy
        df.columns = df.columns.str.strip()

        # Check if the required columns are present in the uploaded CSV
        required_columns = ['public_transport_usage', 'traffic_flow', 
                            'bike_sharing_usage', 'pedestrian_count', 'weather_conditions', 
                            'holiday', 'event', 'temperature', 'humidity', 'road_incidents', 
                            'public_transport_delay', 'bike_availability', 'pedestrian_incidents', 
                            'timestamp', 'required_policemen']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
    else:
        st.error("No file provided or uploaded.")
        return None

    # Drop rows with any null values in the required columns
    df = df.dropna(subset=required_columns)

    # Convert 'timestamp' column to Unix format (seconds since 1970)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %I.%M.%S %p', errors='coerce')
        df['timestamp'] = df['timestamp'].astype(int) // 10**9  # Convert to seconds

    # Convert categorical columns to numeric using LabelEncoder
    le = LabelEncoder()
    df['event'] = le.fit_transform(df['event'].astype(str))
    df['weather_conditions'] = le.fit_transform(df['weather_conditions'].astype(str))
    df['holiday'] = le.fit_transform(df['holiday'].astype(str))

    # Fill missing values for numeric columns with the mean of each numeric column
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Check if the DataFrame is empty after cleaning
    if df.empty:
        st.error(f"All data has been removed during cleaning. Please upload a file with valid data.")
        return None

    return df

# --------- Traffic Optimization ---------

def train_traffic_model(df):
    """
    Train a Random Forest model to predict traffic flow based on features.
    """
    features = ['public_transport_usage', 'bike_sharing_usage', 
                'pedestrian_count', 'temperature', 'humidity', 'road_incidents', 
                'public_transport_delay', 'bike_availability', 'pedestrian_incidents', 
                'event', 'weather_conditions', 'holiday', 'timestamp']
    target = 'traffic_flow'
    
    X = df[features]
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def train_policemen_model(df):
    """
    Train a model to predict the number of policemen required based on traffic flow and other features.
    """
    features = ['traffic_flow', 'temperature', 'humidity', 'road_incidents', 
                'public_transport_delay', 'bike_availability', 'pedestrian_incidents', 
                'event', 'weather_conditions', 'holiday', 'timestamp']
    target = 'required_policemen'
    
    X = df[features]
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def predict_traffic(model, new_data):
    """
    Predict traffic flow for new data using the trained model.
    """
    return model.predict(new_data)

def predict_policemen(model, new_data):
    """
    Predict the number of policemen required based on traffic flow and other factors.
    """
    return model.predict(new_data)

# --------- Emission Reduction ---------

def recommend_emission_reduction(traffic_flow, weather_conditions, public_transport_usage):
    if traffic_flow > 5000 and weather_conditions == 'Clear':
        return "Optimize traffic lights and encourage bike usage."
    elif public_transport_usage > 1000:
        return "Increase public transport frequency."
    else:
        return "Promote carpooling and use eco-friendly vehicles."

# --------- Streamlit Interface ---------

st.title("Smart Traffic Optimization and Emission Reduction System")
st.write("""
    This system predicts traffic flow, recommends emission reduction strategies, 
    and predicts the number of policemen required to optimize traffic.
""")

uploaded_file = st.file_uploader("Upload a CSV file containing traffic data", type="csv")

if uploaded_file is not None:
    data = load_and_process_data(uploaded_file=uploaded_file)

    if data is not None:
        st.write("Processed Data:", data)
        
        traffic_model = train_traffic_model(data)
        policemen_model = train_policemen_model(data)

        if traffic_model and policemen_model:
            st.success("Models trained successfully!")

            # Predict traffic flow
            st.header("Predict Traffic Flow")
            public_transport_users = st.number_input('Public Transport Users (per hour)', min_value=0)
            bike_sharing_usage = st.number_input('Bike Sharing Usage (per hour)', min_value=0)
            pedestrian_count = st.number_input('Pedestrian Count (per hour)', min_value=0)
            temperature = st.number_input('Temperature (°C)', min_value=-50, max_value=50)
            humidity = st.number_input('Humidity (%)', min_value=0, max_value=100)
            road_incidents = st.number_input('Road Incidents (per hour)', min_value=0)
            public_transport_delay = st.number_input('Public Transport Delay (minutes)', min_value=0)
            bike_availability = st.number_input('Bike Availability (per hour)', min_value=0)
            pedestrian_incidents = st.number_input('Pedestrian Incidents (per hour)', min_value=0)

            input_data = {
                'public_transport_usage': [public_transport_users],
                'bike_sharing_usage': [bike_sharing_usage],
                'pedestrian_count': [pedestrian_count],
                'temperature': [temperature],
                'humidity': [humidity],
                'road_incidents': [road_incidents],
                'public_transport_delay': [public_transport_delay],
                'bike_availability': [bike_availability],
                'pedestrian_incidents': [pedestrian_incidents],
                'event': [0], 'weather_conditions': [0], 'holiday': [0], 'timestamp': [0]
            }

            input_df = pd.DataFrame(input_data)

            if st.button('Predict Traffic Flow'):
                traffic_flow_prediction = predict_traffic(traffic_model, input_df)
                st.write(f"Predicted Traffic Flow: {traffic_flow_prediction[0]:.2f} vehicles per hour")

            # Predict number of policemen
            st.header("Predict Number of Policemen Required")
            input_data['traffic_flow'] = [traffic_flow_prediction[0]]

            if st.button('Predict Policemen Required'):
                policemen_prediction = predict_policemen(policemen_model, pd.DataFrame(input_data))
                st.write(f"Predicted Number of Policemen Required: {int(policemen_prediction[0])}")
else:
    st.write("Please upload a CSV file to get started.")
