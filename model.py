#importing necessory libraries
import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("train.csv")

#dropping unnecessery column
data = data.drop(columns=['Loan_ID'])

#filling nan values
data.Gender = data.Gender.fillna('Male')
data.Married = data.Married.fillna('Yes')
data.Dependents = data.Dependents.fillna('0')
data.Self_Employed = data.Self_Employed.fillna('No')
data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
data.Loan_Amount_Term = data.Loan_Amount_Term.fillna(360.0)
data.Credit_History = data.Credit_History.fillna(1.0)

# converting categorical varibale into contnuous variables

# creating dummy variable 
data = pd.get_dummies(data,columns = ['Gender','Married','Education','Self_Employed','Loan_Status'],drop_first=True)

# creating continuous variable of those columns which have more than two unique values 
def convert_property_area(word):
    '''this function converts property area into continuous variable'''
    prop = {'Semiurban':1,'Urban':2,'Rural':3}
    return prop[word]
data['Property_Area'] = data['Property_Area'].apply(lambda x : convert_property_area(x))

def convert_dependents(word):
    '''this function converts dependents into continuous variable'''
    dep = {'0':0,'1':1,'2':2,'3+':3}
    return dep[word]
data['Dependents'] = data['Dependents'].apply(lambda x : convert_dependents(x))

# splitting data into target and independent variables
x = data.drop(columns = 'Loan_Status_Y')
y = data['Loan_Status_Y']

# scaling the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler_x = scaler.fit_transform(x)

# splitting the dataset
from sklearn.model_selection import train_test_split as tts
train_x,test_x,train_y,test_y = tts(scaler_x,y,train_size = 0.8,stratify = y)

# building the logistic regression model

from sklearn.linear_model import LogisticRegression as lr
classifier  = lr(class_weight='balanced')

# fitting model with training data
classifier.fit(train_x,train_y)
predicted_values = classifier.predict(test_x)
predicted_probabilities = classifier.predict_proba(test_x)

# # confusion matrix,accuracy,
# from sklearn.metrics import confusion_matrix
# cf = confusion_matrix(test_y,predicted_values)
# print(cf)

# print(classifier.score(test_x,test_y)
# # generate whole report
# from sklearn.metrics import classification_report as cr
# k = cr(test_y,predicted_values)
# print(k)

file = 'modell.pkl'
fileobj = open(file,'wb')
pickle.dump(classifier,fileobj)
fileobj.close()
