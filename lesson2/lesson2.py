import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from   sklearn               import metrics

PATH     = "../datasets/"
FILE     = "heart_disease.csv"
data     = pd.read_csv(PATH + FILE)
x_data   = data.drop("target", axis=1)
y_values = data["target"]

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_values, test_size=0.3, random_state=42
)

# Stochastic gradient descent models are sensitive to differences
from sklearn.preprocessing import StandardScaler
scaler        = StandardScaler()
scaler.fit(X_train)
X_trainScaled = scaler.transform(X_train)
X_testScaled  = scaler.transform(X_test)

clf     = LogisticRegression(max_iter=1000)
clf.fit(X_trainScaled, y_train)
lr_pred = clf.predict(X_testScaled)

print("Accuracy:{} ".format(clf.score(X_testScaled, y_test) * 100))
print("Error Rate:{} ".format((1 - clf.score(X_testScaled, y_test)) * 100))

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(y_test, lr_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ',metrics.accuracy_score(y_test, lr_pred))
print("\nConfusion Matrix")
print(confusion_matrix)

COLUMN_DIMENSION = 1
#######################################################################
# Part 2
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras.optimizer_v2.adam import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

# shape() obtains rows (dim=0) and columns (dim=1)
n_features = X_trainScaled.shape[COLUMN_DIMENSION]

with tf.device('/cpu:0'):
    def create_model():
        model = Sequential()
        model.add(Dense(5, input_dim=n_features, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
        opt = Adam(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=20, verbose=1)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
    print("Baseline RMSE: " + str(np.sqrt(results.std())))

    model = create_model()
    history = model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=1,
                        validation_data=(X_testScaled, y_test))

    loss, acc = model.evaluate(X_testScaled, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)




