import pandas as pd
import dask.dataframe as dd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify

# --------- Data Preprocessing ---------

def load_and_process_data(file_path):
    # Load the dataset using Dask for handling large files
    df = dd.read_csv(file_path)
    
    # Preprocess the data (handling missing values, types, etc.)
    df = df.dropna()  # Drop missing values
    df['Timestamp'] = dd.to_datetime(df['Timestamp'])  # Convert to datetime
    df['Day of Week'] = df['Timestamp'].dt.dayofweek  # Extract day of week
    
    # Convert categorical columns to category type for optimization
    df['Event'] = df['Event'].astype('category')
    df['Weather Conditions'] = df['Weather Conditions'].astype('category')
    
    # Compute the dataframe (force evaluation of the Dask dataframe)
    df = df.compute()
    
    return df

# --------- Traffic Optimization ---------

def train_traffic_model(df):
    # Train a model to predict traffic flow (no. of vehicles passing a point)
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
    # Predict traffic flow for new data
    return model.predict(new_data)

# --------- Emission Reduction ---------

def recommend_emission_reduction(traffic_flow, weather_conditions, public_transport_usage):
    if traffic_flow > 5000 and weather_conditions == 'Clear':
        return "Optimize traffic lights and encourage bike usage"
    elif public_transport_usage > 1000:
        return "Increase public transport frequency"
    else:
        return "Promote carpooling and use eco-friendly vehicles"

# --------- Web App (Flask) ---------

app = Flask(__name__)

# Load and preprocess data
data = load_and_process_data('data/raw/traffic_data.csv')

# Train the traffic model
traffic_model = train_traffic_model(data)

@app.route('/predict_traffic', methods=['POST'])
def predict_traffic_flow():
    data = request.json  # Receive new data for prediction
    new_data = pd.DataFrame(data)  # Convert received data into DataFrame
    traffic_flow_prediction = predict_traffic(traffic_model, new_data)
    return jsonify({"traffic_flow_prediction": traffic_flow_prediction.tolist()})

@app.route('/recommend_emission_reduction', methods=['POST'])
def emission_reduction():
    data = request.json
    traffic_flow = data['traffic_flow']
    weather_conditions = data['weather_conditions']
    public_transport_usage = data['public_transport_usage']
    
    recommendation = recommend_emission_reduction(traffic_flow, weather_conditions, public_transport_usage)
    return jsonify({"recommendation": recommendation})

if __name__ == "__main__":
    app.run(debug=True)
