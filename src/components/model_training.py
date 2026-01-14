import sys
import pandas as pd
import os 
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from src.utils import evaluate_model
# from src.components.Data_transformation import DataTransformation

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def iniate_model_training(self, train_array, test_array):

        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }

            param_grids = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear"]
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20]
                },
                "Support Vector Machine": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                }
            }
            report, best_models = evaluate_model(X_train, y_train, X_test, y_test,models,param_grids)

            print("Model Evaluation Report:")
            for model_name, accuracy in report.items():
                print(f"{model_name} : {accuracy}")

            print('\n====================================================================================\n')

            logging.info("Model Evaluation Report:")
            for model_name, accuracy in report.items():
                logging.info(f"{model_name} : {accuracy}")





        
        except Exception as e:
            logging.info("Exception accured  while training the model")
            raise CustomException(e,sys)
        



