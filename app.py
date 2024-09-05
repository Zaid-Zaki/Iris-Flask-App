from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

def decision_tree_classifier(input_data):
    # This is a simple example, you can add more conditions or refine the logic based on your trained model
    sepal_length, sepal_width, petal_length, petal_width = input_data

    # Example of basic decision logic (not a real model, replace with actual rules)
    if petal_length < 2.5:
        return 0  # Setosa
    elif petal_width < 1.8:
        return 1  # Versicolor
    else:
        return 2  # Virginica

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    sepal_length = float(data['sepal_length'])
    sepal_width = float(data['sepal_width'])
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])

    # Input data
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Predict the class using the decision tree classifier
    prediction = decision_tree_classifier(input_data)
    
    # Map the class to the iris species
    iris_classes = ['setosa', 'versicolor', 'virginica']
    predicted_class = iris_classes[prediction]

    return jsonify({'class': predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
