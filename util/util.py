from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, make_scorer
import pandas as pd
import numpy as np


def get_pipeline(categorical : list, numeric_features : list, model):
  #this function creates a pipeline for the model, applies necessary transformations
  column_transformer = ColumnTransformer([
      ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
      ('scaling', StandardScaler(), numeric_features)
  ])

  pipeline = Pipeline(steps=[
      ('ohe_and_scaling', column_transformer),
      ('regression', model)
  ])
  return pipeline


def evaluate(predictions:list, test_labels:list): -> float
    #this function return accuracy score for the model and test labels
    accuracy = accuracy_score(test_labels, predictions)
    print('Model Performance')
    print('Accuracy = {:0.5f}%.'.format(accuracy))
    return accuracy


def load_model(filename : str, model):
    #loading the model from pkl file
    model = pickle.load(open(filename, 'rb')) #To load saved model from local directory
    return model
  
  
def save_model(filename : str):
    #dumping the model to the pkl file
    pickle.dump(model, open(filename, 'wb')) #Saving the model
    
