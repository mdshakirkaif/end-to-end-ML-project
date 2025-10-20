import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj

class PredictPipeline:
    def __init__(self):
        pass
    try:
        def predict(self,features):
            model_path='artifacts\model.pkl'
            preprossor_path='artifacts\preprocessor.pkl'
            print("BEFORE LOADING")
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprossor_path)
            print("AFTER LOADING")
            print(preprocessor.feature_names_in_)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
    except Exception as e:
        raise CustomException(e,sys)
                

class customData:
    def __init__(self,gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        self.gender=gender,
        self.race_ethnicity=race_ethnicity,
        self.parental_level_of_education=parental_level_of_education,
        self.lunch=lunch,
        self.test_preparation_course=test_preparation_course,
        self.reading_score=reading_score,
        self.writing_score=writing_score
    
    def get_data_dataframe(self):
        try:
            data={
                "gender":self.gender,
                "race_ethnicity":self.race_ethnicity,
                "parental_level_of_education":self.parental_level_of_education,
                "lunch":self.lunch,
                "test_preparation_course":self.test_preparation_course,
                "reading_score":self.reading_score,
                "writing_score":self.writing_score
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            raise CustomException(e,sys)