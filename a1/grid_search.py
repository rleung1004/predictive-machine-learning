import json

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import RepeatedKFold

from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def grid_search(random: bool = False, filename: str = "grid_search"):
    df = pd.read_csv("./datasets/car_prices.csv")
    print(df.head())
    print(df.tail())

    y = df.copy(True)
    y = y[['current price']]
    x = pd.get_dummies(df, columns=['condition', 'years'])
    x = x.drop(['current price', 'top speed', 'hp', 'torque', 'economy', 'rating',
                'condition_1', 'condition_2', 'condition_3', 'condition_4', 'condition_5'], axis=1)
    print(x.head())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled, y_scaled = x_scaler.fit_transform(x_train), y_scaler.fit_transform(y_train)

    n_features = len(x.columns)

    # ---Model Func---#
    def create_model(num_neurons: int = 5, initializer: str = 'he_normal', num_layers: int = 0, lr: float = 0.001):
        model = Sequential()
        model.add(Dense(num_neurons, kernel_initializer=initializer,
                        input_dim=n_features, activation='relu'))
        for i in range(num_layers):
            model.add(Dense(num_neurons * 2 // 3 + 1, kernel_initializer=initializer,
                            activation='relu'))
        model.add(Dense(1, kernel_initializer=initializer, activation='linear'))

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], optimizer=opt)
        return model

    params = {
        'initializer': ['he_normal', 'he_uniform'],
        'batch_size': [10, 50, 100, 150, 200, 300, 400, 450, 500],
        'epochs': [50, 100, 200, 300, 400, 500],
        'num_neurons': [10, 20, 30, 40, 50, 60, 80, 150],
        'num_layers': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        'lr': [0.001, 0.0005, 0.002]
    }
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = KerasRegressor(build_fn=create_model, verbose=1)
    if random:
        grid = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=-1, cv=cv,
                                  scoring='neg_root_mean_squared_error', n_iter=60)
    else:
        grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=cv,
                            scoring='neg_root_mean_squared_error')

    # ---summarize results---#
    grid_result = grid.fit(x_scaled, y_scaled)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    results = {mean: f'({std:.6f}) {param}' for mean, std, param in zip(means, stds, params)}
    sorted_results = dict(sorted(results.items(), key=lambda x: x[0], reverse=True))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    with open(f'{filename}.json', 'w') as fp:
        json.dump(sorted_results, fp)

    with open(f'{filename}.txt', 'a') as fp:
        for k, v in sorted_results.items():
            fp.write(f'{k:.6f}:{v}\n')


if __name__ == '__main__':
    grid_search(random=True, filename="with_condition")
