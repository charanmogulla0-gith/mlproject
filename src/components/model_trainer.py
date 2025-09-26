import os
import sys
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# External libraries
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Trains multiple models, evaluates them and saves the best one.
        """
        try:
            logging.info("Starting model training...")

            # Define candidate models
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "KNeighbors": KNeighborsRegressor(n_neighbors=5),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42, verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
            }

            model_report = {}
            best_model_name = None
            best_model_score = -np.inf
            best_model = None

            for name, model in models.items():
                logging.info(f"Training model: {name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                logging.info(f"{name} R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")
                model_report[name] = {"r2": r2, "rmse": rmse}

                if r2 > best_model_score:
                    best_model_score = r2
                    best_model_name = name
                    best_model = model

            logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # Save the best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as f:
                pickle.dump(best_model, f)

            logging.info("Saved best model to pkl file")

            return model_report, self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
