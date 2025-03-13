from flask import Flask, render_template, request
import numpy as np
import requests
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("cropNarayangarh_model.pkl")

# Function to fetch real-time weather data
def get_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            #"rainfall": data.get('rain', {}).get('1h', 0)
        }
    else:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    crop_prediction = None
    error_message = None

    if request.method == "POST":
        API_KEY = "2d730e4db19e9f91a79d0efce720e6c9"  # Replace with your actual API key
        city = request.form["city"].strip()
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Pottasium'])

        # Fetch real-time weather data
        weather = get_weather_data(API_KEY, city)

        if weather:
            # Prepare input for prediction (Temperature & Humidity only)
            input_data = np.array([[weather["temperature"], weather["humidity"],ph,rainfall,N,P,K]])
            prediction = model.predict(input_data)

            crop_prediction = f"✅ Recommended Crop for {city}: {prediction[0]} 🌱"
        else:
            error_message = "❌ Unable to fetch weather data. Check city name or API key."

    return render_template("index.html", crop_prediction=crop_prediction, error_message=error_message)

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=10000,debug=True)
    import os
    port = int(os.environ.get("PORT", 5000))  # Azure assigns a dynamic port
    app.run(host="0.0.0.0", port=port)
