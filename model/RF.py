from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer


def split(X,y):
    # Splitting variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    
def gridsearch():
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
    rf = RandomForestClassifier(random_state = 42)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3,
                                   verbose=2, random_state=42, n_jobs = -1, scoring = make_scorer(accuracy_score))

    
def training():
    clf = RandomForestClassifier(n_estimators = estimators, criterion = criterion, max_depth = depth,
                             min_samples_split = min_samples, max_features = max_features, n_jobs = -1, random_state = 42, max_samples = max_samples)
    clf.fit(X_train, y_train)
