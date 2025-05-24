from flask import Flask, render_template, request
import numpy as np
import requests
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("Short_Model.pkl")

# Function to fetch real-time weather data
def get_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
        }
    else:
        return None

# Function to fetch latitude and longitude for geocoding (OpenCage API)
def get_lat_lon(city, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={city}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            lat = data["results"][0]["geometry"]["lat"]
            lon = data["results"][0]["geometry"]["lng"]
            return lat, lon
    return None, None

@app.route("/", methods=["GET", "POST"])
def index():
    crop_prediction = None
    error_message = None

    if request.method == "POST":
        WEATHER_API_KEY = "2d730e4db19e9f91a79d0efce720e6c9"
        GEOCODE_API_KEY = "28ed1e49114f4621a9fde29887b1697a"

        city = request.form["city"].strip()
        ph_value = request.form.get("ph")

        try:
            ph = float(ph_value)
        except (TypeError, ValueError):
            error_message = "❌ Invalid pH value entered."
            return render_template("index.html", error_message=error_message)

        # Fetch weather data
        weather = get_weather_data(WEATHER_API_KEY, city)
        if weather:
            print("Weather data fetched successfully:", weather)
        else:
            print("Failed to fetch weather data")
            error_message = "❌ Unable to fetch weather data."
            return render_template("index.html", error_message=error_message)

        # Fetch latitude and longitude (optional if needed elsewhere)
        lat, lon = get_lat_lon(city, GEOCODE_API_KEY)
        if lat is None or lon is None:
            print("Failed to fetch geolocation")
            error_message = "❌ Unable to fetch geolocation data."
            return render_template("index.html", error_message=error_message)

        print(f"Using user-provided pH: {ph}")

        # Prepare input for prediction
        input_data = np.array([[weather["temperature"], weather["humidity"], ph]])

        try:
            prediction = model.predict(input_data)
            crop_prediction = f"✅ Recommended Crop for {city}: {prediction[0]} 🌱"
        except Exception as e:
            print(f"Error making prediction: {e}")
            error_message = "❌ Unable to make a prediction."
            return render_template("index.html", error_message=error_message)

    return render_template("index.html", crop_prediction=crop_prediction, error_message=error_message)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
