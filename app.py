import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle
import pandas as pd


app = Flask(__name__)


#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    tenth = float(request.args.get('tenth'))
    twelth=float(request.args.get('twelth'))
    btech=float(request.args.get('btech'))
    sevsem=float(request.args.get('7sem'))
    sixsem=float(request.args.get('6sem'))
    fivesem=float(request.args.get('5sem'))
    final=float(request.args.get('final'))
    medium=float(request.args.get('medium'))
    model1=float(request.args.get('model1'))

    if model1==0:
      model=pickle.load(open('project6_decision_model.pkl','rb'))
    elif model1==1:
      model=pickle.load(open('project6_svm.pkl','rb'))
    elif model1==2:
      model=pickle.load(open('project6_random_forest.pkl','rb'))
    elif model1==3:
      model=pickle.load(open('project6_knn.pkl','rb'))
    elif model1==4:
      model=pickle.load(open('project6_naive.pkl','rb'))
      

    dataset= pd.read_excel('DATASET education.xlsx')
    X = dataset.iloc[:, 0:8].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    prediction = model.predict(sc.transform([[tenth,twelth,btech,sevsem,sixsem,fivesem,final,medium]]))
    if prediction==0:
      message="Student not Placed"
    else:
      message="Student will be placed"
    
        
    return render_template('index.html', prediction_text='Model  has predicted : {}'.format(message))


if __name__ == "__main__":
    app.run(debug=True)
