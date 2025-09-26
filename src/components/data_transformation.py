import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, input_df: pd.DataFrame):
        """
        This function creates and returns the ColumnTransformer object.
        """
        try:
            # identify numeric and categorical columns
            numeric_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                                    'lunch', 'test_preparation_course']

            # pipeline for numeric features
            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info(f"Numeric columns: {numeric_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numeric_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data for transformation")

            # define X and y (assuming we want to predict math_score for example)
            target_column = 'math_score'
            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_feature_test = test_df[target_column]

            # get preprocessor object
            preprocessor = self.get_data_transformer_object(train_df)

            logging.info("Fitting preprocessor on training data")
            input_features_train_transformed = preprocessor.fit_transform(input_features_train)

            logging.info("Transforming test data")
            input_features_test_transformed = preprocessor.transform(input_features_test)

            # save the preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logging.info("Saved preprocessor object")

            return (
                input_features_train_transformed,
                input_features_test_transformed,
                target_feature_train,
                target_feature_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)