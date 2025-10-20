import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_obj

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation(self):
        '''This function is responsible for data transformation'''
        
        try:
            num_features=["writing_score", "reading_score"]
            cat_features=[ "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",]
            
            num_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scale',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scale',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            preprocessor=ColumnTransformer(
                [
                    ('num_features',num_pipeline,num_features),
                    ('cat_features',cat_pipeline,cat_features)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initate_data_transformation(self,train_path,test_path):
        
        try:
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            preprocessor_obj=self.get_data_transformation()
            
            target_feature='math_score'
            
            train_data=train.drop(columns=target_feature,axis=1)
            target_train_data=train[target_feature]
            
            test_data=test.drop(columns=target_feature,axis=1)
            target_test_data=test[target_feature]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_features_train_arr=preprocessor_obj.fit_transform(train_data)
            input_features_test_arr=preprocessor_obj.transform(test_data)
            
            train_arr=np.c_[
                input_features_train_arr,np.array(target_train_data)
            ]
            test_arr=np.c_[
                input_features_test_arr,np.array(target_test_data)
            ]
            
            logging.info(f"Saved preprocessing object.")
            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_path,
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)
            

