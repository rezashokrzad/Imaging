# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:30:17 2023

@author: rezas
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_mnist.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST' and 'image' in request.files:
        # Handle the uploaded image
        image = Image.open(request.files['image']).convert('L')  # Convert to grayscale
        image_arr = np.asarray(image)
        image_arr = image_arr.reshape(1, -1)  # Reshape to 1x784


        # Make prediction
        prediction = model.predict(image_arr)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
