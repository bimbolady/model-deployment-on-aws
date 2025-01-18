import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('price optimisation', 'rb'))

# Load the scaler (if used during training)
# scaler = StandardScaler() # Assuming you saved the scaler as well
# scaler = pickle.load(open('scaler.pkl', 'rb')) # Load if you saved the scaler

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure the required features are present in the input data
        required_features = ["Product_ID", "Product_Category", "Price (USD)", "Cost (USD)", "Profit_Margin (%)", "Units_Sold", "Advertising_Spend (USD)", "Competitor_Price (USD)"]
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])

        # Scale the input data using the same scaler used during training
        # input_scaled = scaler.transform(input_df) # Uncomment if you used a scaler

        # Make prediction
        prediction = model.predict(input_df)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
