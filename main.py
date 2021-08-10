from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():   
    return render_template("index.html")

@app.route('/predict', methods = ["GET", "POST"] )
def predict():
    user_gender = request.form.get("user_gender")
    if user_gender == "male":
        gender = 0
    else:
        gender = 1
    age = request.form.get("user_age")
    height = request.form.get("user_height")    
    weight = request.form.get("user_weight")
    duration = request.form.get("user_dur")
    ht_rate = request.form.get("user_heart_rt")
    temp = request.form.get("user_bdy_temp")

   
    with open("calories_burnt_prediction_model","rb") as file:
        model = pickle.load(file)

    input = pd.DataFrame([[gender,age,height,weight,duration,ht_rate,temp]], columns=['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'])

    prediction = model.predict(input)[0]
    if prediction <= 250:
        stat = "can push more!!"
    else:
        stat = "great!"
    return render_template("index.html", prediction_result=prediction, stat = stat)

if __name__ == "__main__":
    app.run(debug=True)
