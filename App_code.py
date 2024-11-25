import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import os

# Function to load data and clean it
def load_data(uploaded_file):
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(uploaded_file)
        
        # Remove rows with any null values
        df.dropna(inplace=True)
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return None

# Function to display data overview
def display_data(df):
    st.subheader('Dataset Overview')
    st.write(df.head())

# Function to perform exploratory data analysis (EDA)
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")
    
    # Summary statistics
    st.write(df.describe())

    # Visualizations
    st.subheader("Traffic Flow vs Public Transport Usage")
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=df['Number of public transport users per hour'], y=df['Traffic flow (no of vehicles passing specific point)'])
    plt.title('Traffic Flow vs Public Transport Usage')
    st.pyplot()

    st.subheader("Traffic Flow vs Temperature")
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=df['Temperature'], y=df['Traffic flow (no of vehicles passing specific point)'])
    plt.title('Traffic Flow vs Temperature')
    st.pyplot()

    st.subheader("Traffic Flow by Day of the Week")
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df['Day of the week'], y=df['Traffic flow (no of vehicles passing specific point)'])
    plt.title('Traffic Flow by Day of the Week')
    st.pyplot()

# Function to train a regression model for prediction
def predict_traffic(df):
    st.subheader("Predict Traffic Flow")
    
    # Select features and target variable for prediction
    features = ['Number of public transport users per hour', 'Bike sharing usage', 'Pedestrian count', 'Temperature', 'Humidity']
    target = 'Traffic flow (no of vehicles passing specific point)'

    # Check if all required columns are present in the dataframe
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}")
        return
    
    X = df[features]
    y = df[target]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    st.write("Model Training Complete")
    st.write(f"Model Coefficients: {model.coef_}")
    st.write(f"Model Intercept: {model.intercept_}")
    
    # Predict traffic flow for the test set
    predictions = model.predict(X_test)
    
    # Display predictions vs actual values
    st.subheader("Predictions vs Actual Values")
    prediction_df = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
    st.write(prediction_df.head())

    # Plotting the predictions vs actual values
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, predictions)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Actual Traffic Flow")
    plt.ylabel("Predicted Traffic Flow")
    st.pyplot()

# Main function to run the Streamlit app
def main():
    st.title('Smart Traffic Optimization and Emission Reduction System')

    # File uploader to upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the data and display basic info
        df = load_data(uploaded_file)
        if df is not None:
            display_data(df)
            perform_eda(df)
            predict_traffic(df)
    
if __name__ == "__main__":
    main()
