from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from util.util import load_model, save_model, get_pipeline, evaluate, read_data, read_settings
from conf import settings, logging


def split():
    # Splitting variables into train and test
    settings = read_settings()
    data = read_data()
    X = data.loc[:, data.columns != settings.target]
    y = data[settings.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    logging.info('split the data')
    return X_train, X_test, y_train, y_test

    
def gridsearch(X_train, y_train, rf, num):
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
        pipeline3 = get_pipeline([''],[X_train.columns.values] , rf_random)
        model3 = pipeline3.fit(X_train, y_train)
        splitter = model3[1].best_params_['splitter']
        criterion = model3[1].best_params_['criterion']
        max_depth = [int(x) for x in np.linspace(round(model3[1].best_params_['max_depth'] * 0.75,0), round(model3[1].best_params_['max_depth'] * 1.25,0), num = 5)] 
        min_samples_split = [model3[1].best_params_['max_depth']-1, model3[1].best_params_['max_depth'] ,model3[1].best_params_['max_depth']+1]
        min_samples_leaf = [model3[1].best_params_['min_samples_leaf']-1, model3[1].best_params_['min_samples_leaf'] ,model3[1].best_params_['min_samples_leaf']+1]
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
        pipeline4 = get_pipeline([''],[X_train.columns.values] , grid_search)
        model4 = pipeline4.fit(X_train, y_train)
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
        pipeline3 = get_pipeline([''],[X_train.columns.values] , rf_random)
        model3 = pipeline3.fit(X_train, y_train)
        loss = model3[1].best_params_['loss']
        criterion = model3[1].best_params_['criterion']
        n_estimators = [int(x) for x in np.linspace(round(model3[1].best_params_['n_estimators'] * 0.9,0), round(model3[1].best_params_['n_estimators'] * 1.1,0), num = 3)] 
        learning_rate = [model3[1].best_params_['learning_rate']-0.03, model3[1].best_params_['learning_rate'] ,model3[1].best_params_['learning_rate']+0.03]
        min_samples_leaf = [model3[1].best_params_['min_samples_leaf']-1, model3[1].best_params_['min_samples_leaf'] ,model3[1].best_params_['min_samples_leaf']+1]
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
        pipeline4 = get_pipeline([''],[X_train.columns.values] , grid_search)
        model4 = pipeline4.fit(X_train, y_train)
    logging.info('got the best estimator')
    return grid_search
    
    
def training():
    settings = read_settings()
    clf = RandomForestClassifier(n_jobs = -1, random_state = 42)
    m2 = GradientBoostingClassifier(random_state = 42)
    X_train, X_test, y_train, y_test = split()
    gridsearch(X_train, y_train, clf, 1)
    gridsearch(X_train, y_train, m2, 2)
    save_model(settings.dt_conf,clf)
    save_model(settings.dt_conf_2,m2)
    logging.info('trained the models')

    
def get_predictions(values, m_num):
    settings = read_settings()
    if m_num = 1:
        try:
            model = load_model(settings.dt_conf)
        except:
            logging.info('need to train the forest')
            training()
            model = load_model(settings.dt_conf)
    else:
        try:
            model = load_model(settings.dt_conf_2)
        except:
            logging.info('need to train the boosting')
            training()
            model = load_model(settings.dt_conf_2)
    return model.predict(values)
    
