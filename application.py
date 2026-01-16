from flask import Flask, render_template, request, send_file,redirect,url_for
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
    return redirect(url_for('predict_datapoint'))

# @app.route('/index')
# def about():
#     return "hello"

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            
            pregnancies=float(request.form.get('pregnancies')),
            glucose = float(request.form.get('glucose')),
            blood_pressure = float(request.form.get('blood_pressure')),
            skin_thickness = float(request.form.get('skin_thickness')),
            insulin = float(request.form.get('insulin')),
            bmi = float(request.form.get('bmi')),
            diabetes_pedigree_function = request.form.get('diabetes_pedigree_function'),
            age= request.form.get('age'),
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        pred = predict_pipeline.predict(final_new_data)

        prediction_value = int(pred[0])
        result = "Diabetic" if prediction_value == 1 else "Not Diabetic"

        return render_template('index.html',final_result=result)
    

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # 1. Get uploaded file
        file = request.files['file']

        if file.filename == "":
            return render_template(
                'index.html',
                batch_message="No file selected"
            )

        # 2. Read CSV
        df = pd.read_csv(file)

        # 3. Run predictions
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(df)

        # 4. Convert predictions to readable labels
        df['Prediction'] = [
            "Diabetic" if int(p) == 1 else "Not Diabetic"
            for p in predictions
        ]

        # 5. Save result CSV
        output_path = "artifacts/batch_predictions.csv"
        os.makedirs("artifacts", exist_ok=True)
        df.to_csv(output_path, index=False)

        # 6. Return success message + enable download
        return render_template(
            'index.html',
            batch_message="Batch prediction completed successfully!",
            download_ready=True
        )

    except Exception as e:
        return render_template(
            'index.html',
            batch_message=f"Error: {str(e)}"
        )
    

@app.route('/download_predictions')
def download_predictions():
    file_path = "artifacts/batch_predictions.csv"
    return send_file(file_path, as_attachment=True)

@app.route('/download/train')
def download_train():
    return send_file('artifacts/train.csv', as_attachment=True)

@app.route('/download/test')
def download_test():
    return send_file('artifacts/test.csv', as_attachment=True)

@app.route('/download/model')
def download_model():
    return send_file('artifacts/model.joblib', as_attachment=True)    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
