from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import datetime
app = Flask(__name__)
model = pickle.load(open('Car_Prediction_Random_Forest_Regression.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year']) #2007
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        Fuel_Type=int(request.form['Fuel_Type'])
        Year= datetime.datetime.now().year - Year
        Seller_Type=int(request.form['Seller_Type'])
        Transmission=int(request.form['Transmission'])
        prediction=model.predict([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type,Seller_Type,Transmission]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_text="Sorry! You Cannot Sell The Car")
        else:
            return render_template('index.html',prediction_text="The Selling Price Of Your Car is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

