from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os
import sys
import joblib


def save_object(file_path, obj):
    try:
        logging.info("extracting the path to save the model..")
        dir_path = os.path.dirname(file_path)

        logging.info("creating the directory inside the artifacts")
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            logging.info("Dumping the model to artifacts folder..")
            joblib.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occured while saving the model")
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param_grids):
    """
    Evaluates multiple classification models using GridSearchCV
    and returns best scores, best estimators and best parameters.
    """
    try:
        report = {}
        best_models = {}
        best_params = {}

        for model_name, model in models.items():
            logging.info(f"Tuning and training model: {model_name}")

            if model_name not in param_grids:
                raise ValueError(f"Parameter grid missing for {model_name}")

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            # Predict Testing data
            y_test_pred = best_model.predict(X_test)

            # Evaluate model
            test_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_score
            best_models[model_name] = best_model
            best_params[model_name] = grid.best_params_

            logging.info(
                f"{model_name} | Best Params: {grid.best_params_} | "
                f"Test Accuracy: {test_score}"
            )

        return report, best_models, best_params

    except Exception as e:
        logging.info("exception occured while evaluating the model accuracy_score")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
