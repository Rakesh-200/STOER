import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib  # for caching the trained model

# --------- Data Preprocessing ---------

@st.cache
def load_and_process_data(uploaded_file=None):
    """
    Load and process the traffic data from the uploaded file.
    Uses Dask for large file handling and caching.
    """
    if uploaded_file is not None:
        # Read the file as a Dask DataFrame to handle large files
        ddf = dd.read_csv(uploaded_file)
        df = ddf.compute()  # Convert to pandas dataframe after processing
        st.write(f"Successfully loaded uploaded CSV file.")
        
        # Strip any leading/trailing spaces in column names for accuracy
        df.columns = df.columns.str.strip()

        # Check if the required columns are present in the uploaded CSV
        required_columns = ['public_transport_usage', 'traffic_flow', 
                            'bike_sharing_usage', 'pedestrian_count', 'weather_conditions', 
                            'holiday', 'event', 'temperature', 'humidity', 'road_incidents', 
                            'public_transport_delay', 'bike_availability', 'pedestrian_incidents']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
    else:
        st.error("No file provided or uploaded.")
        return None

    # Drop rows with any null values in the required columns
    df = df.dropna(subset=required_columns)

    # Remove the timestamp column (since we're no longer considering it)
    df = df.drop(columns=['timestamp'], errors='ignore')

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

@st.cache  # Cache the trained model to avoid retraining on each run
def train_traffic_model(df):
    """
    Train a Random Forest model to predict traffic flow based on features.
    Use n_jobs=-1 to speed up training by using multiple cores.
    """
    # Features and target variable
    features = ['public_transport_usage', 'bike_sharing_usage', 
                'pedestrian_count', 'temperature', 'humidity', 'road_incidents', 
                'public_transport_delay', 'bike_availability', 'pedestrian_incidents', 
                'event', 'weather_conditions', 'holiday']
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
    
    # Train a Random Forest model (with parallel processing)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model

def predict_traffic(model, new_data):
    """
    Predict traffic flow for new data using the trained model.
    """
    return model.predict(new_data)

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
                'holiday': [0]  # Assuming default or encoded value for 'holiday'
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
