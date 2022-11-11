# Run this file (do not run "applctn.py")

import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('RegMdl.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
    # after the above block of code is executed, the output is given in Postman. Postman ensures that our project is absolutely working.


# below block of code is not a must to be enabled (commented out) for Postman to return response
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    # return render_template("home.html",prediction_text="The House price prediction is {}".format(output)) #this line returns negative number, so used absolute [abs()] in next line
    return render_template("home.html",prediction_text="The House price prediction is {}".format(abs(output)))



if __name__=="__main__":
    app.run(debug=True)