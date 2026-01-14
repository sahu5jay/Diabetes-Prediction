import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_preprocessor_obj(self):

        try:
            logging.info("Data transformation initated")

            df = pd.read_csv(os.path.join('artifacts','raw.csv'))
            logging.info('raw Data converted to data frame from artifacts')

            num_standard = ['Glucose', 'BloodPressure', 'BMI']
            num_robust = ['Pregnancies', 'SkinThickness', 'Insulin', 'Age']

            zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)

            standard_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            robust_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                ('std', standard_pipeline, num_standard),
                ('rob', robust_pipeline, num_robust)
            ])

            logging.info("returning the preprocessor pipeline")
            return preprocessor
        except FileNotFoundError as e:
            logging.info("FileNotFoundError raised inside preprocessor function")
            raise CustomException(e,sys)
    
    def initate_data_transformation(self,train_path, test_path):

        try:

            logging.info("fetch the path og the train and test data ")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('train and test data is converted to data frame')

            preprocessor_obj = self.get_preprocessor_obj()
            logging.info("preprocessor object collected")

            target_col = 'Outputs'
            drop_col = [target_col]

            logging.info("seperated input feature and target feature from train dataset")
            input_feature_train_df = train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df = train_df[target_col]

            logging.info("seperated input feature and target feature from test dataset")
            input_feature_test_df = test_df.drop(columns=target_col,axis=1)
            target_feature_test_df = test_df[target_col]


            logging.info("fit transform on the trining data set")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)

            logging.info("transform on the test data set")
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("concatinating training feature and target data set ")
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            logging.info("concatinating test feature and target data set ")
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            logging.info("saving the preprocessor file to")
            save_object(
                file_path = self.data_transformation.preprocessor_obj_file,
                obj = preprocessor_obj
            )

        
            logging.info("returning the trainarr, test arr preprocessor pickel file")
            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file
            )

        except FileNotFoundError as e:
            logging('File note found exception under initate_data_transformation function')
            raise CustomException(e, sys)
