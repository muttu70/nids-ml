from flask import Flask, render_template, url_for, request
import sqlite3
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
warnings.filterwarnings('ignore')
# from keras.models import load_model
# from feature import 
from feature import FeatureExtraction


file = open("model/model.pkl","rb")
gbc = pickle.load(file)
file.close()


connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if result:
            return render_template('home.html')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/IDS', methods=['GET', 'POST'])
def IDS():
    if request.method == 'POST':
        url = request.form['Link']
        print(url)
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 
        feature_details = obj.getFeatureDetails()

        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        
        pred = "It is {0:.2f} % safe to Use  ".format(y_pro_phishing*100)
        print(f"\n\n{pred}\n\n")
        
        return render_template('ml.html', xx=round(y_pro_non_phishing, 2), res=round(y_pro_non_phishing, 2), url=url, pred=pred, feature_details=feature_details)
    
    return render_template("ml.html", xx=-1, msg="You are in ML page")

# @app.route('/IDS', methods=['GET', 'POST'])
# def IDS():
#     if request.method == 'POST':
#         url = request.form['Link']
#         print(url)
#         obj = FeatureExtraction(url)
#         x = np.array(obj.getFeaturesList()).reshape(1,30) 

#         y_pred =gbc.predict(x)[0]
#         #1 is safe       
#         #-1 is unsafe
#         y_pro_phishing = gbc.predict_proba(x)[0,0]
#         y_pro_non_phishing = gbc.predict_proba(x)[0,1]
#         # if(y_pred ==1 ):
#         pred = "It is {0:.2f} % safe to Use  ".format(y_pro_phishing*100)
#         print(f"\n\n{pred}\n\n")
#         return render_template('ml.html',xx =round(y_pro_non_phishing,2) , res =round(y_pro_non_phishing,2),url=url, pred=pred )
#     return render_template("ml.html", xx =-1, msg="You are in ML page")




@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
