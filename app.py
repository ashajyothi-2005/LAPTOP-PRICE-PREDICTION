from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
def load_model():
    model_path = "laptop_price_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model file 'laptop_price_model.pkl' not found."

    try:
        # 1. Capture and convert data from the form
        # It is crucial that these keys match the columns in your training CSV
        input_data = {
            "brand_name": [request.form['brand']],
            "processor_brand": [request.form['processor_brand']],
            "processor_type": [request.form['processor_type']],
            "generations": [request.form['generation']],
            "ram": [int(request.form['ram'])],
            "storage_capacity_gb": [int(request.form['storage'])],
            "have_ssd": [int(request.form['ssd'])],
            "have_hdd": [int(request.form['hdd'])],
            "graphics_capacity": [int(request.form['gpu'])],
            "display_size_inch": [float(request.form['screen'])],
            "display_type": [request.form['display']]
        }

        # 2. Create the DataFrame
        df = pd.DataFrame(input_data)

        # 3. Predict using your trained data
        prediction = model.predict(df)[0]
        
        # 4. Format the output price (Indian Rupees)
        formatted_price = "{:,.2f}".format(round(float(prediction), 2))

        # 5. Redirect to the result page with the dynamic price
        return redirect(url_for('show_result', price=formatted_price))
    
    except Exception as e:
        # This will tell you exactly what is wrong if it fails
        return f"Prediction Error: {str(e)}"

@app.route('/result')
def show_result():
    price = request.args.get('price', '0.00')
    return render_template("result.html", price=price)

if __name__ == "__main__":
    app.run(debug=True)