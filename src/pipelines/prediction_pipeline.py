import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.joblib')
            model_path=os.path.join('artifacts','model.joblib')

            logging.info("collected the path of the pickel file")

            preprocessor=load_object(preprocessor_path)
            logging.info("preprocessor pickle file loaded")
            model=load_object(model_path)
            logging.info("model pickle file loaded")

            logging.info("Custom data scaling started")
            data_scaled=preprocessor.transform(features)
            
            logging.info("Custom data scaled")

            pred=model.predict(data_scaled)
            logging.info("returing the predicted price application")
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:

    def __init__(self,
                pregnancies:int,
                glucose:float,
                blood_pressure:float,
                skin_thickness:float,
                insulin:float,
                bmi:float,
                diabetes_pedigree_function:float,
                age:int):
        
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.blood_pressure = blood_pressure
        self.skin_thickness = skin_thickness
        self.insulin = insulin
        self.bmi = bmi
        self.diabetes_pedigree_function = diabetes_pedigree_function
        self.age = age

    def get_data_as_dataframe(self):
        try:
            logging.info("convertig the custom data to dictionary...")
            custom_data_input_dict = {
                'pregnancies':[self.pregnancies],
                'glucose':[self.glucose],
                'blood_pressure':[self.blood_pressure],
                'skin_thickness':[self.skin_thickness],
                'insulin':[self.insulin],
                'bmi':[self.bmi],
                'diabetes_pedigree_function':[self.diabetes_pedigree_function],
                'age':[self.age]
            }
            logging.info("custom data converted to dictionary")
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("returning the data frame to application")
            return df
        except Exception as e:
            logging.info("exception occured at prediction_pipeline in data to dataframe")
            raise CustomException(e,sys)

        


        
    