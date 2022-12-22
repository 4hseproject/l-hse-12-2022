from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from utils import load_model, save_model, get_pipeline, evaluate, read_data, read_settings
from conf import settings


def split(X,y):
    # Splitting variables into train and test
    settings = read_settings()
    data = read_data()
    X = data.loc[:, data.columns != settings.target]
    y = data[settings.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    
def gridsearch(rf, num):
    if num ==1:
        # Number of trees in random forest
        splitter = ['best','random']
        # Number of features to consider at every split
        criterion = ['gini', 'entropy']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Create the random grid
        random_grid = {'splitter': splitter,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'criterion': criterion}
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3,
                                       verbose=2, random_state=42, n_jobs = -1, scoring = make_scorer(accuracy_score))
        splitter = rf_random[1].best_params_['splitter']
        criterion = rf_random[1].best_params_['criterion']
        max_depth = [int(x) for x in np.linspace(round(rf_random[1].best_params_['max_depth'] * 0.75,0), round(rf_random[1].best_params_['max_depth'] * 1.25,0), num = 5)] 
        min_samples_split = [rf_random[1].best_params_['max_depth']-1, rf_random[1].best_params_['max_depth'] ,rf_random[1].best_params_['max_depth']+1]
        min_samples_leaf = [rf_random[1].best_params_['min_samples_leaf']-1, rf_random[1].best_params_['min_samples_leaf'] ,rf_random[1].best_params_['min_samples_leaf']+1]
        # Create the parameter grid based on the results of random search 
        param_grid = {
            'splitter': [splitter],
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_split,
            'min_samples_split': min_samples_leaf,
            'criterion': [criterion]
        }
        # Instantiate the grid search model

        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2, scoring = make_scorer(accuracy_score))
    else:
        # Number of trees in random forest
        loss = ['log_loss','exponential','deviance']
        # Number of features to consider at every split
        criterion = ['friedman_mse', 'squared_error']
        # Maximum number of levels in tree
        n_estimators = [int(x) for x in np.linspace(50, 600, num = 7)]
        # Minimum number of samples required to split a node
        learning_rate = [0.05, 0.1, 0.5, 1,2]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Create the random grid
        random_grid = {'loss': loss,
                   'max_depth': max_depth,
                   'n_estimators': n_estimators,
                       'learning_rate':learning_rate,
                   'min_samples_leaf': min_samples_leaf,
                   'criterion': criterion}
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3,
                                       verbose=2, random_state=42, n_jobs = -1, scoring = make_scorer(accuracy_score))
        loss = rf_random[1].best_params_['loss']
        criterion = rf_random[1].best_params_['criterion']
        n_estimators = [int(x) for x in np.linspace(round(rf_random[1].best_params_['n_estimators'] * 0.9,0), round(rf_random[1].best_params_['n_estimators'] * 1.1,0), num = 3)] 
        learning_rate = [rf_random[1].best_params_['learning_rate']-0.03, rf_random[1].best_params_['learning_rate'] ,rf_random[1].best_params_['learning_rate']+0.03]
        min_samples_leaf = [rf_random[1].best_params_['min_samples_leaf']-1, rf_random[1].best_params_['min_samples_leaf'] ,rf_random[1].best_params_['min_samples_leaf']+1]
        # Create the parameter grid based on the results of random search 
        param_grid = {
            'loss': [loss],
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'min_samples_split': min_samples_leaf,
            'criterion': [criterion]
        }
        # Instantiate the grid search model

        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2, scoring = make_scorer(accuracy_score))
    return grid_search
    
    
def training():
    settings = read_settings()
    clf = RandomForestClassifier(n_jobs = -1, random_state = 42)
    m2 = GradientBoostingClassifier(random_state = 42)
    gridsearch(clf, 1)
    gridsearch(m2, 2)
    save_model(settings.dt_conf,clf)
    save_model(settings.dt_conf_2,m2)

    
def get_predictions():
    settings = read_settings()
    model = load_model(settings.dt_conf)
    
