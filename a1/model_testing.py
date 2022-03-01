import json

import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

# from eda_v1 import my_get_dummies, view_ols_regression_model

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def my_get_dummies(df: pd.DataFrame, cols: list = None, drop_first: bool = False,
                   drop_orig: bool = False) -> pd.DataFrame:
    cols = df.columns.tolist() if cols is None else cols
    for col in cols:
        dummies = pd.get_dummies(df[[col]], columns=[col], drop_first=drop_first)
        df = pd.concat(([df, dummies]), axis=1) if not drop_orig \
            else pd.concat(([df, dummies]), axis=1).drop([col], axis=1)
    return df


def view_ols_regression_model(x, y, test_split=0.2, print_summary=True, random_state: int = None):
    x = sm.add_constant(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)
    if print_summary: print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    return y_test.values, predictions.values.reshape(-1, 1)


def get_data():
    return pd.read_csv("./datasets/car_prices.csv")


def create_model(**kwargs):
    model = Sequential()
    model.add(Dense(kwargs['num_neurons'], kernel_initializer=kwargs['initializer'],
                    input_dim=kwargs['input_dim'], activation='relu'))
    for i in range(kwargs['num_layers']):
        model.add(Dense(kwargs['num_neurons'], kernel_initializer=kwargs['initializer'],
                        activation='relu'))
    model.add(Dense(1, kernel_initializer=kwargs['initializer'], activation='linear'))

    opt = tf.keras.optimizers.Adam(learning_rate=kwargs['lr'])
    model.compile(loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], optimizer=opt)
    return model


def evaluate_model(x: pd.DataFrame, y: pd.DataFrame, model_params: dict):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled, y_scaled = x_scaler.fit_transform(x_train), y_scaler.fit_transform(y_train)

    model = create_model(**model_params)
    history = model.fit(x_scaled, y_scaled, verbose=1,
                        epochs=model_params['epochs'], batch_size=model_params['batch_size'])

    predictions = model.predict(x_scaler.transform(x_test))
    unscaled_pred = y_scaler.inverse_transform(predictions)
    mse_scaled = metrics.mean_squared_error(y_test, unscaled_pred)

    view_ols_regression_model(x, y)
    print(f"ANN RMSE scaled: {np.sqrt(mse_scaled)}")


def evaluate_model_1():
    x, y = get_model_1_data()
    print(x.head())

    model_params = {'num_neurons': 60, 'num_layers': 8, 'lr': 0.0005,
                    'initializer': 'he_uniform', 'epochs': 400,
                    'batch_size': 10, 'input_dim': len(x.columns)}
    evaluate_model(x, y, model_params)


def get_model_1_data():
    df = get_data()
    y = df.copy(True)
    y = y[['current price']]
    x = my_get_dummies(df, cols=['condition', 'rating'], drop_orig=True)
    x = x.drop(labels=['economy', 'current price', 'top speed', 'hp', 'torque', 'condition_1', 'condition_2',
                       'condition_3', 'condition_4', 'rating_1', 'rating_2', 'rating_3'], axis=1)
    return x, y


if __name__ == '__main__':
    evaluate_model_1()
