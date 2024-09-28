import pandas as pd
import numpy as np
from django.shortcuts import render
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Load the car prices dataset (this should be moved to a model or stored in DB for production)
data = pd.read_csv("car_prices.csv")

# Data preprocessing (same as your script)
data.rename({"Unnamed: 0.1": "a"}, axis="columns", inplace=True)
data.drop(["a"], axis=1, inplace=True)
data.rename({"Unnamed: 0": "b"}, axis="columns", inplace=True)
data.drop(["b"], axis=1, inplace=True)
data['stroke'] = data['stroke'].fillna(data['stroke'].mean())

# Features and target
car_features = ["engine-size", 'horsepower', 'city-mpg', 'highway-mpg', "bore", "stroke", "peak-rpm", "normalized-losses", 'symboling', "wheel-base", "length", 'height', 'width', "curb-weight"]
X = data[car_features]
y = data['price']

# Train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

def predict_car_price(request):
    if request.method == 'POST':
        # Get user input from form
        input_data = {
            'engine-size': float(request.POST['engine-size']),
            'horsepower': float(request.POST['horsepower']),
            'city-mpg': float(request.POST['city-mpg']),
            'highway-mpg': float(request.POST['highway-mpg']),
            'bore': float(request.POST['bore']),
            'stroke': float(request.POST['stroke']),
            'peak-rpm': float(request.POST['peak-rpm']),
            'normalized-losses': float(request.POST['normalized-losses']),
            'symboling': int(request.POST['symboling']),
            'wheel-base': float(request.POST['wheel-base']),
            'length': float(request.POST['length']),
            'height': float(request.POST['height']),
            'width': float(request.POST['width']),
            'curb-weight': float(request.POST['curb-weight']),
        }
        input_df = pd.DataFrame([input_data])

        # Make prediction
        predicted_price = model.predict(input_df)[0]

        # Pass the prediction to template
        return render(request, 'predict.html', {'predicted_price': predicted_price})

    return render(request, 'predict.html')

def visualize_data(request):
    # Create correlation heatmap
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')

    # Save plot to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the image to display in the HTML template
    image_b64 = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'visualize.html', {'image': image_b64})
