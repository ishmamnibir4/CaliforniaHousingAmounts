# This file is just for explanation. Do not run it! RUn the file "application.py"

import json
import pickle
# in requirement.txt, we have already installed all the imported libraries in this file
# we use this pickle file to load the regression model

# from flask import Flask,request,app,jsonify,url_for,render_template,redirect,flash,session,escape is what he [initailly] wrote
# 'Flask' library. 'flask' is basically used to create lightweight web applications

from flask import Flask,request,app,jsonify,url_for,render_template
# above line: cause he said we will use till 'render_template'

import numpy as NP
import pandas as PD

# [below code] first step: we create our 'app' (keyword already existing). We created basic 'flask' app
# [below code] "__name__" is the starting point of our application from where it will run
app = Flask(__name__)

# [below code]: we load the pickle file [Loading the model]
# in the 'open function', we have the name of the file to load. Opening the file in readbyte (rb) mode and loading it
Regr_model = pickle.load(open('RegMdl.pkl','rb'))

scalar = pickle.load*(open('scaling.pkl','rb')) # for importing

# in Flask, we create the app.route
# this will have the first root. We can say "the local host, the url and slash, if we say, we should go to the homepage"
@app.route('/')

# [below code] "home.html" will be the html welcome-page
# by default, when we hit this flask app, we will be redirected to home.html
def HomePageFunc():
    return render_template('home.html') #this will return the html page
    # just above line: "render_template" will look at a template folder, so we created the "templates" folder, inside which we then create 'home.html'




# [below]: POST request cause we are giving an input. Then we will give this input to our model, and the model will give the output
@app.route('/predict_api',methods=['POST']) #now we create a predict API. We will create an API, where, using "Postman" or any other tool, we can send the request to our app, and can get the output
# [below: will help us predict]
# as soon we hit the api "'/predict_api'" as a post request for "'dataa'" information, whatever info is present inside "'dataa'", we will capture is using request.json 
def predict_api(): 
    dataa=request.json['dataa']
    print(dataa)
    print(NP.array(list(dataa.values())).reshape(1,-1)) # to get the dictionary values. If we get these values and convert it to list, then we get a single list
    # just above line: purpose of reshape with 1,-1 -> we are saying that this is the single data point recorded, that we will get, and we will use this for the transformation. 
    # we will now pass the same thing on 'scaling.pkl'
    new_dataa = scalar.transform(NP.array(list(dataa.values())).reshape(1,-1))
    output=Regr_model.predict(new_dataa)
    print(output[0]) # output in 2d array
    return jsonify(output[0])

# then, in the LnrReg.ipynb file, we did the standardization after the train test split. During this standardization, we forgot to import or transform this particular standardization into a pickle file
# we will use this same thing in this file as well.
# We go to the execute LnrReg.ipynb till that part, and we will create a pickle file 'pickle.dump(std_scaling,open('scaling.pkl','wb'))'
# then we back here



# to run this:
if __name__ == "__main__":
    app.run(debug=True)


# http://127.0.0.1:5000 is the server, and we created API "predict_api"
# if we run "http://127.0.0.1:5000/predict_api", the webpage says "Method Not Allowed. The method is not allowed for the requested URL."
# this is cause the method is "POST". We must give some information using 'postman'
# googled 'postman download for windows' downloaded and installed it.
# See the word document 'Postman-related'

