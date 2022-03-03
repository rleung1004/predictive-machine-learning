import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pickle as pkl


def save_scalers(x_scaler=None, y_scaler=None, is_manual: bool = True):
    if x_scaler is None and y_scaler is None:
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
    if is_manual:
        pkl.dump(x_scaler, open('manual_feature_set_x_scaler.pkl', 'wb'))
        pkl.dump(y_scaler, open('manual_feature_set_y_scaler.pkl', 'wb'))
    else:
        pkl.dump(x_scaler, open('selected_feature_set_x_scaler.pkl', 'wb'))
        pkl.dump(y_scaler, open('selected_feature_set_y_scaler.pkl', 'wb'))


def load_scalers(is_manual: bool):
    if is_manual:
        x_scaler = pkl.load(open('manual_feature_set_x_scaler.pkl', 'rb'))
        y_scaler = pkl.load(open('manual_feature_set_y_scaler.pkl', 'rb'))
    else:
        x_scaler = pkl.load(open('selected_feature_set_x_scaler.pkl', 'rb'))
        y_scaler = pkl.load(open('selected_feature_set_y_scaler.pkl', 'rb'))
    return x_scaler, y_scaler


def prepare_model_data():
    df = pd.read_csv("datasets/car_prices.csv")
    y = df[['current price']]
    x = pd.get_dummies(df, columns=['condition', 'rating', 'economy', 'years'])
    x = x.drop(['current price'], axis=1)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # return x_scaled, x_test_scaled, y_scaled, y_test
    return train_test_split(x, y, test_size=0.2)


def view_stopping_plots(figname, history, model):
    # Plot loss learning curves.
    plt.subplot(211)
    plt.title(f'{model}Loss as MSE and RMSE by Epoch', pad=-40)
    plt.plot(history.history['loss'][:150], label='Loss')
    plt.plot(history.history['root_mean_squared_error'][:150], label='RMSE')
    plt.legend()
    # Plot accuracy learning curves.
    plt.subplot(212)
    plt.title(f'{model}Loss as MSE and RMSE by Epoch', pad=-40)
    plt.plot(history.history['loss'][150:], label='Loss')
    plt.plot(history.history['root_mean_squared_error'][150:], label='RMSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{figname}')
    plt.clf()


def evaluateModel(model, x_test, y_test, model_name, is_manual: bool):
    print(f"***** {model_name} *****")
    predictions = model.predict(x_test)
    x_scaler, y_scaler = load_scalers(is_manual)
    unscaled_pred = y_scaler.inverse_transform(predictions)
    rmse = np.sqrt(mean_squared_error(y_test, unscaled_pred))
    print('Root Mean Squared Error:', rmse)
    with open('model_evaluation.txt', 'a') as file:
        file.write(f"{model_name} RMSE: {rmse}\n")


def create_model(**kwargs):
    model = Sequential()
    model.add(Dense(kwargs['num_neurons'], kernel_initializer=kwargs['initializer'],
                    input_dim=kwargs['input_dim'], activation='relu'))
    for i in range(kwargs['num_layers']):
        model.add(Dense(kwargs['num_neurons'] * 2 // 3 + 1, kernel_initializer=kwargs['initializer'],
                        activation='relu'))
    model.add(Dense(1, kernel_initializer=kwargs['initializer'], activation='linear'))

    opt = tf.keras.optimizers.Adam(learning_rate=kwargs['lr'])
    model.compile(loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], optimizer=opt)
    return model


def early_stopping(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                   model_params: dict, filename: str = "model", model_name: str = None, is_manual: bool = True):
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100)
    mc = ModelCheckpoint(f'{filename}.h5', monitor='loss', mode='min', verbose=1,
                         save_best_only=True)

    model = create_model(**model_params)
    history = model.fit(x_train, y_train, verbose=1,
                        epochs=model_params['epochs'], batch_size=model_params['batch_size'],
                        callbacks=[es, mc])

    print(f"Epoch Stopped: {es.stopped_epoch}")
    view_stopping_plots(filename, history, model_name if model_name else "")
    evaluateModel(model, x_test, y_test, model_name, is_manual)
    return model


def build_manual_feature_set_models(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame,
                                    y_test: pd.DataFrame):
    feature_list = ['on road old', 'on road now', 'km', 'condition_6', 'condition_7', 'condition_8', 'condition_9',
                    'condition_10', 'years_2', 'years_3', 'years_4', 'years_5', 'years_6', 'years_7']

    model_x_train = x_train.copy(True)[feature_list]
    model_x_test = x_test.copy(True)[feature_list]
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled, y_scaled = x_scaler.fit_transform(model_x_train), y_scaler.fit_transform(y_train.copy(True))
    x_test_scaled = x_scaler.fit_transform(model_x_test)
    save_scalers(x_scaler, y_scaler, is_manual=True)

    model_params_list = [
        {'num_neurons': 6, 'num_layers': 2, 'lr': 0.002, 'initializer': 'he_normal', 'batch_size': 10, 'epochs': 2000,
         'input_dim': len(model_x_train.columns.tolist())},
        {'num_neurons': 7, 'num_layers': 2, 'lr': 0.00075, 'initializer': 'he_uniform', 'batch_size': 10,
         'epochs': 2000, 'input_dim': len(model_x_train.columns.tolist())},
    ]

    models = []
    for i, model_params in enumerate(model_params_list):
        models.append(early_stopping(x_scaled, x_test_scaled, y_scaled, y_test, model_params,
                                     f"./models/manual_feature_set_model_{i}", f"Manual Feature Set Model {i}",
                                     is_manual=True))

    return models


def build_selected_best_feature_set_models(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame,
                                           y_test: pd.DataFrame):
    feature_list = ['on road old', 'on road now', 'km', 'top speed', 'hp', 'torque', 'condition_1', 'condition_2',
                    'condition_3', 'condition_4', 'condition_5', 'condition_6', 'condition_8', 'condition_9',
                    'condition_10', 'rating_1', 'rating_2', 'rating_4', 'rating_5', 'economy_8', 'economy_9',
                    'economy_12', 'economy_13', 'economy_14', 'economy_15', 'years_3', 'years_4', 'years_5',
                    'years_6', 'years_7']

    model_x_train = x_train.copy(True)[feature_list]
    model_x_test = x_test.copy(True)[feature_list]
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled, y_scaled = x_scaler.fit_transform(model_x_train), y_scaler.fit_transform(y_train.copy(True))
    x_test_scaled = x_scaler.fit_transform(model_x_test)
    save_scalers(x_scaler, y_scaler, is_manual=False)

    model_params_list = [
        {'num_neurons': 7, 'num_layers': 2, 'lr': 0.002, 'initializer': 'he_uniform', 'batch_size': 10, 'epochs': 2000,
         'input_dim': len(model_x_train.columns.tolist())},
        {'num_neurons': 6, 'num_layers': 2, 'lr': 0.0015, 'initializer': 'he_normal', 'batch_size': 10, 'epochs': 2000,
         'input_dim': len(model_x_train.columns.tolist())},
    ]

    models = []
    for i, model_params in enumerate(model_params_list):
        models.append(early_stopping(x_scaled, x_test_scaled, y_scaled, y_test, model_params,
                                     f"./models/selected_feature_set_model_{i}", f"Selected Feature Set Model {i}",
                                     is_manual=False))

    return models


def build_stacked_model(models: list = None):
    if models is None:
        print("No base models are given")
        return


def main():
    x_train, x_test, y_train, y_test = prepare_model_data()
    models = []

    manual_models = build_manual_feature_set_models(x_train, x_test, y_train, y_test)
    for model in manual_models:
        models.append(model)

    select_k_best_models = build_selected_best_feature_set_models(x_train, x_test, y_train, y_test)
    for model in select_k_best_models:
        models.append(model)


if __name__ == '__main__':
    main()
