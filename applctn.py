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
# We go to the execute LnrReg.ipynb till that part, and we will create a pickle file 'pickle.dump(std_scaling,open('scaling.pkl','wb'))', then we come back here
# above block of code: So far, we have created an API (predict_api), which was a post request, and we collected the data using Postman, and based on this, we were giving some kind of output after passing it through scalar transformation (that is standardization). And from that we also passed to through the regression 
# model (which is basically giving the output)

# [below block of code]: instead of creating in the form of API, why don't we just create a small web application, where we provide the inputs, then submit the form (as soon
# as we submit the form, we take the data here, and we do the prediction with the help of our model)
@app.route('/predict',methods=['POST']) # /predict is a new function
def predict():# then we define our 'definition predict'. This is obviously a POST method
# below code: let's say we will have an html page, and from the html fields, we will try to give all the input values from a specific form. So we will create a form, which
# will (probably) require to give inputs from the user, and those inputs will be all the features from the file "LnrReg.ipynb". We will give all these features in html
# form, and from this form, we will try to get the data in application.py (it will be a post request)
    data=[float(x) for x in request.form.values()] # whatever values we are filling in this form, we can capture it cause all the info will be present in this request object. We convert into float value cause all the values are to be inputted as float wrt the model. For loop:  every "x" in the request.form will be converted into float, and finally we get in the form of the list format
    final_input = scalar.transform(NP.array(data).reshape(1,-1)) # whatever value there is inside "data", we have to reshape it
    print(final_input) #to see the output of our model
    output = regmodel.predict(final_input)[0] # without '[0]', we would get array of two dimensions, so with '[0]' we take out our first value
    return render_template("home.html") # render_template is very important in flask! Here we render a certain html. We redirect it to "home.html". In "home.html", we keep a placeholder called prediction_text. So, after we get the output, we render the 'home.html' and it will replace the placeholder 'prediction_text', and we replace "{}" with our received output

# to run this:
if __name__ == "__main__":
    app.run(debug=True)


# http://127.0.0.1:5000 is the server, and we created API "predict_api"
# if we run "http://127.0.0.1:5000/predict_api", the webpage says "Method Not Allowed. The method is not allowed for the requested URL."
# this is cause the method is "POST". We must give some information using 'postman'
# googled 'postman download for windows' downloaded and installed it.
# See the word document 'Postman-related'

