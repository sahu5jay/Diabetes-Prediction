import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.joblib")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def iniate_model_training(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
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

            report, best_models, best_params = evaluate_model(
                X_train, y_train, X_test, y_test, models, param_grids
            )

            print("Model Evaluation Report:")
            for model_name, accuracy in report.items():
                print(f"{model_name} : {accuracy}")

            print('\n====================================================================================\n')

            logging.info("Model Evaluation Report:")
            for model_name, accuracy in report.items():
                logging.info(f"{model_name} : {accuracy}")

            # Get best model score
            best_model_score = max(report.values())

            # Get best model name
            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]

            # Get best tuned model & parameters
            best_model = best_models[best_model_name]
            best_model_param = best_params[best_model_name]

            print(
                f"Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}"
            )
            print(f"Best Parameters : {best_model_param}")
            print('\n====================================================================================\n')

            logging.info(
                f"Best Model Found , Model Name : {best_model_name} , Accuracy : {best_model_score}"
            )
            logging.info(f"Best Parameters : {best_model_param}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Exception accured  while training the model")
            raise CustomException(e, sys)
