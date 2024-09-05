from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load and train the model
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Routes
@app.route('/')
def home():
    return render_template('index.html'),500

@app.route('/predict', methods=['POST'])
def predict():

    data = request.form
    sepal_length = float(data['sepal_length'])
    sepal_width = float(data['sepal_width'])
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])

 
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    
  
    prediction = clf.predict(input_data)
    predicted_class = iris.target_names[prediction[0]]

    return jsonify({'class': predicted_class}),500

if __name__ == "__main__":
    app.run(debug=True)
