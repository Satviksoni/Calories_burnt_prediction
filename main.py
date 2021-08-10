from flask import Flask, render_template, request
# from matplotlib.pyplot import prism
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    iris = load_iris()
    model = DecisionTreeRegressor(random_state=0)
    X_train,X_test,Y_train,Y_test = train_test_split(iris.data,iris.target)
    model.fit(X_train,Y_train)
    pickle.dump(model,open("calories_burnt_prediction_model.pkl","wb"))

    return render_template("index.html")

@app.route('/predict', methods = ["GET", "POST"] )
def predict():
    gender = request.form["user_gender"]
    age = request.form["user_age"]
    height = request.form["user_height"]
    weight = request.form["user_weight"]
    duration = request.form["user_dur"]
    ht_rate = request.form["user_heart_rt"]
    temp = request.form["user_bdy_temp"]

    form_array = np.array([[gender,age,height,weight,duration,ht_rate,temp]])
    model = pickle.load(open("calories_burnt_prediction_model.pkl","rb"))
    prediction = model.predict(form_array)
    # prediction = model.predict([[1,53,173.0,68,7,96.0,39.3]])
    result = prediction[0]

    return render_template("index.html", prediction_result = result )

if __name__ == "__main__":
    app.run(debug=True)
