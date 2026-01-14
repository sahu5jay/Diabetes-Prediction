import os 
from src.components.Data_ingestion import DataIngestion 

if __name__ == '__main__':
    obj = DataIngestion()
    sub,sub2 = obj.initiate_data_ingestion()
    print(sub,sub2)