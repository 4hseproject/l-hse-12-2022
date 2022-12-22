from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, make_scorer


def get_pipeline(categorical, numeric_features, model):
  column_transformer = ColumnTransformer([
      ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
      ('scaling', StandardScaler(), numeric_features)
  ])

  pipeline = Pipeline(steps=[
      ('ohe_and_scaling', column_transformer),
      ('regression', model)
  ])
  return pipeline


def evaluate(predictions, test_labels):
    
    accuracy = accuracy_score(test_labels, predictions)
    print('Model Performance')
    print('Accuracy = {:0.5f}%.'.format(accuracy))
    
    return accuracy


def load_model(filename):
    #loading the model from pkl file
    model = pickle.load(open(filename, 'rb')) #To load saved model from local directory
    return model
def save_model(filename):
    #dumping the model to the pkl file
    pickle.dump(model, open(filename, 'wb')) #Saving the model
    
