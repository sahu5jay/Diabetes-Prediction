from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
from src.pipelines.prediction_pipeline import CustomData
from src.pipelines.prediction_pipeline import PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    # Renders the index.html file from the 'templates' folder
    return render_template('index.html')

# @app.route('/index')
# def about():
#     return "hello"

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            pregnancies=float(request.form.get('pregnancies')),
            glucose = float(request.form.get('glucose')),
            table = float(request.form.get('blood_pressure')),
            blood_pressure = float(request.form.get('skin_thickness')),
            insulin = float(request.form.get('insulin')),
            bmi = float(request.form.get('bmi')),
            diabetes_pedigree_function = request.form.get('diabetes_pedigree_function'),
            age= request.form.get('age'),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('index.html',final_result=results)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
