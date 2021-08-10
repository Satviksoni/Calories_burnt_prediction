from flask import Flask, render_template, request
# from matplotlib.pyplot import prism
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    
    # model = DecisionTreeRegressor(random_state=0)
    # X_train,X_test,Y_train,Y_test = train_test_split(iris.data,iris.target)
    # model.fit(X_train,Y_train)
    # pickle(open("calories_burnt_prediction_mode","rb"))

   

    return render_template("index.html")

@app.route('/predict', methods = ["GET", "POST"] )
def predict():
    gender = request.form.get("user_gender")
    age = request.form.get("user_age")
    height = request.form.get("user_height")    
    weight = request.form.get("user_weight")
    duration = request.form.get("user_dur")
    ht_rate = request.form.get("user_heart_rt")
    temp = request.form.get("user_bdy_temp")

    # form_array = np.array([[gender,age,height,weight,duration,ht_rate,temp]])
   
    with open("calories_burnt_prediction_model","rb") as file:
        model = pickle.load(file)

    input = pd.DataFrame([[gender,age,height,weight,duration,ht_rate,temp]], columns=['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp'])
    # prediction = model.predict([[1,53,173.0,68,7,96.0,39.3]])

    prediction = model.predict(input)
    # print(prediction , "prediction")
    # prediction = model.predict([[gender,age,height,weight,duration,ht_rate,temp]])
    # print(prediction , "prediction")
    # result = prediction[0]
    return render_template("index.html", prediction_result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
