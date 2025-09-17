import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, flash,session
import numpy as np
import pandas as pd

app=Flask(__name__)
# load model
model = pickle.load(open('dtmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])  #to create predict api using postman or anyother tool to send request to our app to get the output
# here it will be POST request bcoz i will give input that will get capture as input to our model and then model give output
def predict_api(): 
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()] # running loop for every value inside this request.form convert into float and finally to get in form of list format
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html",prediction_text = "The cancer prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
