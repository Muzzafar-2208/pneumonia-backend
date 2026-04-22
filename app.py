from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return "Pneumonia API is running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from frontend
        data = request.json
        input_dict = data["features"]

        # Convert to DataFrame
        df = pd.DataFrame([input_dict])

        # Convert categorical values (Yes/No, Male/Female, etc.)
        df = pd.get_dummies(df)

        # Match training columns
        model_columns = model.feature_names_in_
        df = df.reindex(columns=model_columns, fill_value=0)

        # Prediction
        prediction = model.predict(df)[0]

        # Response
        return jsonify({
            "result": "Pneumonia" if prediction == 1 else "Normal"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)