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
        required_columns = ['timestamp', 'public_transport_usage', 'traffic_flow', 
                            'bike_sharing_usage', 'pedestrian_count', 'weather_conditions', 'day_of_week', 
                            'holiday', 'event', 'temperature', 'humidity', 'road_incidents', 
                            'public_transport_delay', 'bike_availability', 'pedestrian_incidents']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
    else:
        st.error("No file provided or uploaded.")
        return None

    # Store the number of rows before cleaning
    initial_row_count = df.shape[0]

    # Preprocess the data (handling types, etc.)
    try:
        # Specify the exact format of the timestamp (DD-MM-YYYY hh.mm.ss AM/PM)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %I.%M.%S %p', errors='coerce')
    except Exception as e:
        st.error(f"Error while parsing timestamps: {str(e)}")
        return None

    # Fill missing or invalid timestamps with the mean of valid timestamps
    invalid_timestamp_count = df['timestamp'].isnull().sum()
    
    if invalid_timestamp_count > 0:
        st.warning(f"There are {invalid_timestamp_count} invalid or missing timestamps. These will be filled with the mean timestamp.")
        
        # Calculate the mean timestamp (excluding invalid ones)
        valid_timestamps = df['timestamp'].dropna()
        mean_timestamp = valid_timestamps.mean()
        
        # Fill invalid timestamps with the mean timestamp
        df['timestamp'].fillna(mean_timestamp, inplace=True)

    # Extract day of the week
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Extract day of the week
    
    # Convert categorical columns to numeric using LabelEncoder
    le = LabelEncoder()
    df['event'] = le.fit_transform(df['event'].astype(str))
    df['weather_conditions'] = le.fit_transform(df['weather_conditions'].astype(str))
    df['holiday'] = le.fit_transform(df['holiday'].astype(str))

    # Fill missing values with the mean of each column (for numerical columns)
    df.fillna(df.mean(), inplace=True)

    # Store the number of rows after cleaning
    final_row_count = df.shape[0]

    # Check if the DataFrame is empty after cleaning
    if final_row_count == 0:
        st.error(f"All data has been removed during cleaning. Please upload a file with valid data.")
        return None
    elif final_row_count < initial_row_count:
        st.warning(f"{initial_row_count - final_row_count} rows were removed due to missing or invalid data.")

    return df

# --------- Traffic Optimization ---------

def train_traffic_model(df):
    """
    Train a Random Forest model to predict traffic flow based on features.
    """
    # Features and target variable
    features = ['public_transport_usage', 'bike_sharing_usage', 
                'pedestrian_count', 'temperature', 'humidity', 'road_incidents', 
                'public_transport_delay', 'bike_availability', 'pedestrian_incidents', 
                'event', 'weather_conditions', 'holiday', 'day_of_week']
    target = 'traffic_flow'
    
    # Select features and target
    X = df[features]
    y = df[target]
    
    # Check for missing values (again) in case any are left after preprocessing
    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        st.error("There are still missing values in the features or target variable.")
        return None

    # Ensure all data is numeric (this step is more robust after encoding)
    X = X.apply(pd.to_numeric, errors='coerce')
    y = y.apply(pd.to_numeric, errors='coerce')
    
    # Check if there is data for training
    if X.shape[0] == 0:
        st.error("No data available for training. Please ensure there are enough valid rows in your dataset.")
        return None
    
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

        # Proceed only if model training is successful
        if traffic_model is not None:
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
                'public_transport_usage': [public_transport_users],
                'bike_sharing_usage': [bike_sharing_usage],
                'pedestrian_count': [pedestrian_count],
                'temperature': [temperature],
                'humidity': [humidity],
                'road_incidents': [road_incidents],
                'public_transport_delay': [public_transport_delay],
                'bike_availability': [bike_availability],
                'pedestrian_incidents': [pedestrian_incidents],
                'event': [0],  # Assuming default or encoded value for 'event'
                'weather_conditions': [0],  # Assuming default or encoded value for 'weather_conditions'
                'holiday': [0],  # Assuming default or encoded value for 'holiday'
                'day_of_week': [0]  # Default day of week
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
