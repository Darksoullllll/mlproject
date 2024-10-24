# converting the categorical data into numerical data
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artificats','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
    
    # doing data tranformation
    def get_data_transformation_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            # Creating pipeline for numerical features
            num_pipeline = Pipeline(
                #Pipeline those these steps
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            # Creating a Pipeling for Categorical features
            cat_pipeline = Pipeline(
                #Pipeline those these steps
                steps = [
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("standard_scaling",StandardScaler(with_mean=False))

                ]
            )

            logging.info("Numerical Pipeline Completed")
            logging.info("Categorical Pipeline Completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_features),
                    ("categorical_pipline",cat_pipeline,categorical_features)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train , test is completed")

            logging.info("obtaining preprocessor object")


            preprocessor_obj  = self.get_data_transformation_object()

            target_col_name = "math_score"
            numerical_features = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying the preprocessor object to train and test dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr= np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            