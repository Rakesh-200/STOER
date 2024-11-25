import streamlit as st
import pandas as pd
import vaex
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import traceback

try:
    # Your app code here (e.g., loading data, visualizations)
    st.title("Smart Traffic Optimization and Emission Reduction System")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the file and process
        st.write("File uploaded successfully!")
        # Add the rest of your code here
except Exception as e:
    st.error("An error occurred. Please check the logs.")
    st.write(traceback.format_exc())

# Set page configuration for Streamlit
st.set_page_config(page_title="Smart Traffic Optimization and Emission Reduction", layout="wide")

# Title of the App
st.title("Smart Traffic Optimization and Emission Reduction System")

# File Upload
uploaded_file = st.file_uploader("Upload your data (CSV file)", type=["csv"])

def load_data(uploaded_file):
    """Function to load large data efficiently using vaex"""
    if uploaded_file is not None:
        # Load using vaex for large CSVs (it handles memory efficiently)
        try:
            df = vaex.from_csv(uploaded_file, convert=True)
            st.write(f"Data Loaded: {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    return None

# Load the data if a file is uploaded
data = load_data(uploaded_file)

# Check if data is loaded successfully
if data is not None:
    # Display raw data preview
    st.subheader("Raw Data Preview")
    st.write(data.head())

    # Data Cleaning: Removing rows with null values
    data_cleaned = data.dropna()
    st.write(f"After removing null values: {data_cleaned.shape[0]} rows left.")
    
    # Display cleaned data preview
    st.subheader("Cleaned Data Preview")
    st.write(data_cleaned.head())

    # Traffic Visualization: Plot traffic flow
    st.subheader("Traffic Flow Visualization")
    traffic_data = data_cleaned['Traffic Flow'].to_numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(traffic_data)
    plt.title("Traffic Flow Over Time")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Traffic Flow (No of Vehicles)")
    st.pyplot(plt)

    # Weather Data Visualization: Plot temperature and humidity
    st.subheader("Weather Data Visualization")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(data_cleaned['Temperature'], color='tab:blue')
    ax[0].set_title("Temperature Over Time")
    ax[0].set_xlabel("Time (Hours)")
    ax[0].set_ylabel("Temperature (°C)")

    ax[1].plot(data_cleaned['Humidity'], color='tab:orange')
    ax[1].set_title("Humidity Over Time")
    ax[1].set_xlabel("Time (Hours)")
    ax[1].set_ylabel("Humidity (%)")

    st.pyplot(fig)

    # Traffic Optimization Model: Predict Traffic Flow Based on Weather Conditions
    st.subheader("Traffic Optimization Model: Predicting Traffic Flow")

    # Extract features and target variable for modeling
    features = ['Temperature', 'Humidity', 'Public Transport Usage', 'Bike Usage', 'Pedestrian Count']
    target = 'Traffic Flow'

    # Check if all required columns are present
    if all(col in data_cleaned.columns for col in features + [target]):
        X = data_cleaned[features].to_pandas_df()  # Convert to pandas for scikit-learn
        y = data_cleaned[target].to_pandas_df()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the traffic flow on the test set
        y_pred = model.predict(X_test)

        # Calculate and display the model performance
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE) of the Model: {mse:.2f}")

        # Display prediction vs actual values
        st.subheader("Prediction vs Actual Traffic Flow")
        df_pred = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
        st.write(df_pred.head(10))

    else:
        st.error("Required columns for modeling are missing. Please check your dataset.")
else:
    st.info("Please upload a dataset to start.")

