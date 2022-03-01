from sklearn.model_selection import cross_val_score
from mlxtend.classifier      import EnsembleVoteClassifier
from xgboost                 import XGBClassifier, plot_importance
from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import datasets
import pandas as pd

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def exercise_2():
    iris = datasets.load_iris()
    data = pd.DataFrame({
        'sepal length': iris.data[:, 0],
        'sepal width': iris.data[:, 1],
        'petal length': iris.data[:, 2],
        'petal width': iris.data[:, 3],
        'species': iris.target
    })
    X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
    y = data['species']

    ada_boost   = AdaBoostClassifier()
    grad_boost  = GradientBoostingClassifier()
    xgb_boost   = XGBClassifier()
    boost_array = [ada_boost, grad_boost, xgb_boost]
    eclf        = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost,
                                               xgb_boost], voting='hard')
    labels = ['Ada Boost', 'Grad Boost', 'XG Boost', 'Ensemble']

    for clf, label in zip([ada_boost, grad_boost, xgb_boost, eclf], labels):
        scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
        print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(),
                                                              scores.std(), label))


def example_2():
    from pydataset import data
    import pandas as pd

    # Get the housing data
    df = data('Housing')

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n*** Before data prep.")
    print(df.head(5))

    # Convert continues price variable into evenly distributed categories.
    df['price'] = pd.qcut(df['price'], 3, labels=[0, 1, 2]).cat.codes

    # Show new price variable cateogories.
    print("\nNewly categorized target (price) values")
    print(df['price'].value_counts())

    def convertToBinaryValues(df, columns):
        for i in range(0, len(columns)):
            df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
        return df

    df = convertToBinaryValues(df, ['driveway', 'recroom', \
                                    'fullbase', 'gashw', 'airco', 'prefarea'])

    # Split into two sets
    y = df['price']
    X = df.drop('price', 1)

    # Show prepared data.
    print("\n*** X")
    print(X.head(5))

    print("\n*** y")
    print(y.head(5))

    from sklearn.ensemble import BaggingClassifier, \
        ExtraTreesClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # Create classifiers
    rf = RandomForestClassifier()
    et = ExtraTreesClassifier()
    knn = KNeighborsClassifier()
    svc = SVC()
    rg = RidgeClassifier()
    lr = LogisticRegression(fit_intercept=True, solver='liblinear')

    # Build array of classifiers.
    classifierArray = [rf, et, knn, svc, rg, lr]

    def showStats(classifier, scores):
        print(classifier + ":    ", end="")
        strMean = str(round(scores.mean(), 2))

        strStd = str(round(scores.std(), 2))
        print("Mean: " + strMean + "   ", end="")
        print("Std: " + strStd)

    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    def evaluateModel(model, X_test, y_test, title):
        print("\n*** " + title + " ***")
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        recall = metrics.recall_score(y_test, predictions, average='weighted')
        precision = metrics.precision_score(y_test, predictions, average='weighted')
        f1 = metrics.f1_score(y_test, predictions, average='weighted')

        print("Accuracy:  " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall:    " + str(recall))
        print("F1:        " + str(f1))

    # Search for the best classifier.
    for clf in classifierArray:
        modelType = clf.__class__.__name__

        # Create and evaluate stand-alone model.
        clfModel = clf.fit(X_train, y_train)
        evaluateModel(clfModel, X_test, y_test, modelType)

        # max_features means the maximum number of features to draw from X.
        # max_samples sets the percentage of available data used for fitting.
        bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=3,
                                        n_estimators=10)
        baggedModel = bagging_clf.fit(X_train, y_train)
        evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)


def exercise_6():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    PATH = "../datasets/"
    df = pd.read_csv(PATH + 'loan_default.csv')
    print(df.head())

    # Prepare the data.
    X = df.copy()
    del X['default']
    y = df[['default']]

    from sklearn.ensemble import BaggingClassifier, \
        ExtraTreesClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # Create classifiers
    rf = RandomForestClassifier()
    et = ExtraTreesClassifier()
    knn = KNeighborsClassifier()
    svc = SVC()
    rg = RidgeClassifier()
    lr = LogisticRegression(fit_intercept=True, solver='liblinear')

    # Build array of classifiers.
    classifierArray = [rf, et, knn, svc, rg, lr]

    def showStats(classifier, scores):
        print(classifier + ":    ", end="")
        strMean = str(round(scores.mean(), 2))

        strStd = str(round(scores.std(), 2))
        print("Mean: " + strMean + "   ", end="")
        print("Std: " + strStd)

    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    def evaluateModel(model, X_test, y_test, title):
        print("\n*** " + title + " ***")
        predictions = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        recall = metrics.recall_score(y_test, predictions, average='weighted')
        precision = metrics.precision_score(y_test, predictions, average='weighted')
        f1 = metrics.f1_score(y_test, predictions, average='weighted')

        print("Accuracy:  " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall:    " + str(recall))
        print("F1:        " + str(f1))

    # Search for the best classifier.
    for clf in classifierArray:
        modelType = clf.__class__.__name__

        # Create and evaluate stand-alone model.
        clfModel = clf.fit(X_train, y_train)
        evaluateModel(clfModel, X_test, y_test, modelType)

        # max_features means the maximum number of features to draw from X.
        # max_samples sets the percentage of available data used for fitting.
        bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=3,
                                        n_estimators=10)
        baggedModel = bagging_clf.fit(X_train, y_train)
        evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)


def example_4():
    import pandas as pd
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Load and prepare data.
    FOLDER = '../datasets/'
    FILE = 'petrol_consumption.csv'
    dataset = pd.read_csv(FOLDER + FILE)
    print(dataset)
    X = dataset.copy()
    del X['Petrol_Consumption']
    y = dataset[['Petrol_Consumption']]

    # Create random split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def evaluateModel(model, X_test, y_test, title):
        print("\n****** " + title)
        predictions = model.predict(X_test)
        print('Root Mean Squared Error:',
              np.sqrt(mean_squared_error(y_test, predictions)))

    # Build linear regression ensemble.
    ensembleModel = BaggingRegressor(base_estimator=LinearRegression(), max_features=4,
                                     max_samples=0.5,
                                     n_estimators=10).fit(X_train, y_train)
    evaluateModel(ensembleModel, X_test, y_test, "Ensemble")

    # Build stand alone linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    evaluateModel(model, X_test, y_test, "Linear Regression")


def example_5_grid_search():
    import pandas as pd
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Load and prepare data.
    FOLDER = '../datasets/'
    FILE = 'petrol_consumption.csv'
    dataset = pd.read_csv(FOLDER + FILE)
    print(dataset)
    X = dataset.copy()
    del X['Petrol_Consumption']
    y = dataset[['Petrol_Consumption']]

    feature_combo_list = []

    def evaluateModel(model, X_test, y_test, title, num_estimators, max_features):
        print("\n****** " + title)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Store statistics and add to list.
        stats = {"type": title, "rmse": rmse,
                 "estimators": num_estimators, "features": max_features}
        feature_combo_list.append(stats)

    num_estimator_list = [750, 800, 900, 1000]
    max_features_list = [2, 3, 4]
    max_samples_list = [0.4, 0.5, 0.75, 0.8]

    for num_estimators in num_estimator_list:
        for max_features in max_features_list:
            for max_samples in max_samples_list:
                # Create random split.
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                # Build linear regression ensemble.
                ensembleModel = BaggingRegressor(base_estimator=LinearRegression(),
                                                 max_features=max_features,
                                                 max_samples=max_samples,
                                                 n_estimators=num_estimators).fit(X_train, y_train)
                evaluateModel(ensembleModel, X_test, y_test, "Ensemble",
                              num_estimators, max_features)

                # Build stand alone linear regression model.
                model = LinearRegression()
                model.fit(X_train, y_train)
                evaluateModel(model, X_test, y_test, "Linear Regression", None, None)

    # Build data frame with dictionary objects.
    dfStats = pd.DataFrame()
    print(dfStats)
    for combo in feature_combo_list:
        dfStats = dfStats.append(combo, ignore_index=True)

    # Sort and show all combinations.
    # Show all rows
    pd.set_option('display.max_rows', None)
    dfStats = dfStats.sort_values(by=['type', 'rmse'])
    print(dfStats)


def example_6_stacked_models():
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Prep data.
    PATH = "../datasets/"
    CSV_DATA = "winequality.csv"
    dataset = pd.read_csv(PATH + CSV_DATA)
    X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']].values
    y = dataset['quality']

    def getUnfitModels():
        models = list()
        models.append(ElasticNet())
        models.append(SVR(gamma='scale'))
        models.append(DecisionTreeRegressor())
        models.append(AdaBoostRegressor())
        models.append(RandomForestRegressor(n_estimators=10))
        models.append(ExtraTreesRegressor(n_estimators=10))
        return models

    def evaluateModel(y_test, predictions, model):
        mse = mean_squared_error(y_test, predictions)
        rmse = round(np.sqrt(mse), 3)
        print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)

    def fitBaseModels(X_train, y_train, X_test, models):
        dfPredictions = pd.DataFrame()

        # Fit base model and store its predictions in dataframe.
        for i in range(0, len(models)):
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_test)
            colName = str(i)
            # Add base model predictions to column of data frame.
            dfPredictions[colName] = predictions
        return dfPredictions, models

    def fitStackedModel(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Get base models.
    unfitModels = getUnfitModels()

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
    stackedModel = fitStackedModel(dfPredictions, y_test)

    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_val)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluateModel(y_val, predictions, models[i])

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\n** Evaluate Stacked Model **")
    evaluateModel(y_val, stackedPredictions, stackedModel)


example_6_stacked_models()
