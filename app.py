# importing necessary libraries
import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)


obj = pickle.load(open('modell.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features  =[np.array(features)]
    prediction = obj.predict(final_features)

    return render_template('index.html',prediction_text='Loan will be approved :{}'.format(prediction))


if __name__ == "__main__":
    app.run(debug = True)