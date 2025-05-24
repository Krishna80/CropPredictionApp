from flask import Flask, render_template, request, url_for
import numpy as np
import requests
import joblib
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("Short_Model.pkl")
le = joblib.load("label_encoder.pkl")  # Load the label encoder

# Get weather using latitude and longitude
def get_weather_data(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
        }
    else:
        return None

# Get location details using reverse geocoding (OpenCage)
def get_location_details(lat, lon, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={lat}+{lon}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            # Return full components dictionary and formatted address
            components = data["results"][0]["components"]
            formatted = data["results"][0]["formatted"]
            return components, formatted
        except (KeyError, IndexError):
            return None, "Unknown Location"
    return None, "Unknown Location"

# Check if location is allowed (not water body or building)
def is_location_allowed(components):
    if not components:
        return False

    disallowed_keywords = [
        "sea", "ocean", "pond", "lake", "river", "water",
        "building", "residential", "house", "hotel", "parking", "airport"
    ]

    for key, value in components.items():
        if isinstance(value, str):
            val_lower = value.lower()
            for bad_word in disallowed_keywords:
                if bad_word in val_lower:
                    return False
    return True

@app.route("/", methods=["GET", "POST"])
def index():
    crop_prediction = None
    error_message = None
    latitude = None
    longitude = None

    if request.method == "POST":
        WEATHER_API_KEY = "2d730e4db19e9f91a79d0efce720e6c9"
        GEOCODE_API_KEY = "28ed1e49114f4621a9fde29887b1697a"

        lat = request.form.get("latitude")
        lon = request.form.get("longitude")
        ph_value = request.form.get("ph")

        # Check required fields
        if not lat or not lon or not ph_value:
            error_message = "❌ All fields are required. Please select a location and provide soil pH."
            return render_template("index.html", error_message=error_message, latitude=lat, longitude=lon)

        try:
            lat = float(lat)
            lon = float(lon)
            ph = float(ph_value)
            latitude = lat
            longitude = lon
        except (TypeError, ValueError):
            error_message = "❌ Invalid input. Latitude, longitude, and pH must be numeric."
            return render_template("index.html", error_message=error_message, latitude=None, longitude=None)

        # Validate value ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            error_message = "❌ Coordinates are out of valid range."
            return render_template("index.html", error_message=error_message, latitude=None, longitude=None)

        if not (3.0 <= ph <= 10.0):
            error_message = "❌ Soil pH should be between 3.0 and 10.0."
            return render_template("index.html", error_message=error_message, latitude=latitude, longitude=longitude)

        # Get location details for validation
        components, place_name = get_location_details(lat, lon, GEOCODE_API_KEY)

        if not is_location_allowed(components):
            error_message = "❌ Selected location appears to be a water body, building, or invalid for agriculture. Please select a valid location."
            return render_template("index.html", error_message=error_message, latitude=latitude, longitude=longitude)

        # Fetch weather data
        weather = get_weather_data(WEATHER_API_KEY, lat, lon)
        if not weather:
            error_message = "❌ Unable to fetch weather data. Please try a different location."
            return render_template("index.html", error_message=error_message, latitude=latitude, longitude=longitude)

        # Prepare input for model
        input_data = np.array([[weather["temperature"], weather["humidity"], ph]])

        try:
            pred_numeric = model.predict(input_data)
            pred_label = le.inverse_transform(pred_numeric)[0]  # Convert to original label
            plant_icon = '<img src="' + url_for('static', filename='crop_pic.png') + '" alt="plant icon" style="width: 20px; vertical-align: middle;">'
            crop_prediction = f"✅ Recommended Crop for {place_name}: {pred_label} {plant_icon}"
        except Exception as e:
            print(f"Error making prediction: {e}")
            error_message = "❌ Unable to make a prediction at this time."

    return render_template(
        "index.html",
        crop_prediction=crop_prediction,
        error_message=error_message,
        latitude=latitude,
        longitude=longitude
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
