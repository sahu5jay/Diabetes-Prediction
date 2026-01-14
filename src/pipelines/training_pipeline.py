import os 
from src.components.Data_ingestion import DataIngestion 
from src.components.Data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging

if __name__ == '__main__':
    obj = DataIngestion()
    sub,sub2 = obj.initiate_data_ingestion()

    obj2 = DataTransformation()
    train_arr,test_arr,_=obj2.initate_data_transformation(sub,sub2)
    print(train_arr,test_arr)