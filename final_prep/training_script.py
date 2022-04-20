import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy import arange
from sklearn import metrics
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pickle as pkl

FEATURE_LIST = ['on road old', 'on road now', 'km', 'condition_6', 'condition_7', 'condition_8', 'condition_9',
                       'condition_10', 'years_2', 'years_3', 'years_4', 'years_5', 'years_6', 'years_7']


def save_scalers(x_scaler=None, y_scaler=None):
    if x_scaler is None and y_scaler is None:
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
    pkl.dump(x_scaler, open('./src/x_scaler.pkl', 'wb'))
    pkl.dump(y_scaler, open('./src/y_scaler.pkl', 'wb'))


def load_scalers():
    x_scaler = pkl.load(open('./src/x_scaler.pkl', 'rb'))
    y_scaler = pkl.load(open('./src/y_scaler.pkl', 'rb'))
    return x_scaler, y_scaler


def prepare_model_data():
    df = pd.read_csv("../datasets/fluDiagnosis.csv")
    y = df[['Diagnosed']]
    x = df.drop(['Diagnosed'], axis=1)
    return train_test_split(x, y, test_size=0.2)


def view_stopping_regression_plots(figname, history, model):
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


def view_stopping_classifcation_plots(figname, history, model):
    # Plot loss learning curves.
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 1)
    # Visualize loss history for training data.
    plt.plot(epoch_count, training_loss, label='Train Loss', color='red')

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
             color='black')

    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.title(f'{model}Loss by Epoch', pad=-40)

    # Plot accuracy learning curves.
    # Get training and test loss histories
    training_loss = history.history['accuracy']
    validation_loss = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)
    # Visualize loss history for training data.
    plt.plot(epoch_count, training_loss, label='Train Accuracy', color='red')

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, 'r--',
             label='Validation Accuracy', color='black')
    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.title(f'{model}Accuracy by Epoch', pad=-40)

    plt.tight_layout()
    plt.savefig(f'{figname}')
    plt.clf()


def evaluateRegressionModel(model, x_test, y_test, model_name):
    print(f"***** {model_name} *****")
    predictions = model.predict(x_test)
    x_scaler, y_scaler = load_scalers()
    unscaled_pred = y_scaler.inverse_transform(predictions)
    rmse = np.sqrt(mean_squared_error(y_test, unscaled_pred))
    print('Root Mean Squared Error:', rmse)
    with open('model_evaluation.txt', 'a') as file:
        file.write(f"{model_name} RMSE: {rmse}\n")


def evaluateClassicationModel(model, x_test, y_test, model_name):
    print(f"***** {model_name} *****")
    predictions = model.predict(x_test)
    # Show confusion matrix and accuracy scores.
    matrix = metrics.confusion_matrix(y_test, np.rint(predictions))
    accuracy = metrics.accuracy_score(y_test, np.rint(predictions))
    print('\nAccuracy: ', accuracy)
    print("\nConfusion Matrix")
    print(matrix)
    with open('model_evaluation.txt', 'a') as file:
        file.write(f"{model_name} Accuracy: {accuracy}")


def create_model(**kwargs):
    model = Sequential()
    # input layer
    model.add(Dense(kwargs['num_neurons'], kernel_initializer=kwargs['initializer'],
                    input_dim=kwargs['input_dim'], activation='relu'))
    # hidden layers
    for i in range(kwargs['num_layers']):
        model.add(Dense(kwargs['num_neurons'] * 2 // 3 + 1, kernel_initializer=kwargs['initializer'],
                        activation='relu'))
    # output layer
    model.add(Dense(1, kernel_initializer=kwargs['initializer'], activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=kwargs['lr'])
    model.compile(loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy()], optimizer=opt)
    return model


def early_stopping(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                   model_params: dict, filename: str = "model", model_name: str = None):
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=75)
    mc = ModelCheckpoint(f'{filename}.h5', monitor='loss', mode='min', verbose=1,
                         save_best_only=True)

    model = create_model(**model_params)
    history = model.fit(x_train, y_train, verbose=1,
                        epochs=model_params['epochs'], batch_size=model_params['batch_size'],
                        callbacks=[es, mc], validation_data=(x_test, y_test))

    print(f"Epoch Stopped: {es.stopped_epoch}")
    view_stopping_classifcation_plots(filename, history, model_name if model_name else "")
    evaluateClassicationModel(model, x_test, y_test, model_name)
    return model


def build_stand_alone_models(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame,
                             y_test: pd.DataFrame):
    model_x_train = x_train.copy(True)
    model_x_test = x_test.copy(True)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    # x_scaled, y_scaled = x_scaler.fit_transform(model_x_train), y_scaler.fit_transform(y_train.copy(True))
    x_scaled = x_scaler.fit_transform(model_x_train)
    x_test_scaled = x_scaler.fit_transform(model_x_test)
    save_scalers(x_scaler, y_scaler)

    model_params_list = [
        {'num_neurons': 100, 'num_layers': 1, 'lr': 0.01, 'initializer': 'he_uniform', 'batch_size': 5,
         'epochs': 100, 'input_dim': len(model_x_train.columns.tolist())},
    ]

    models = []
    for i, model_params in enumerate(model_params_list):
        models.append(early_stopping(x_scaled, x_test_scaled, y_train, y_test, model_params,
                                     f"./src/ann_model_{i}", f"ANN Model {i}"))

    return models


def build_stacked_model(models: list, x_test: pd.DataFrame, x_val: pd.DataFrame,
                        y_test: pd.DataFrame, y_val: pd.DataFrame):
    df_predictions = pd.DataFrame()
    df_validation_predictions = pd.DataFrame()
    for i, model in enumerate(models):
        x_scaler, y_scaler = load_scalers()
        x_test_scaled = x_scaler.transform(x_test.copy(True))
        x_val_scaled = x_scaler.transform(x_val.copy(True))
        predictions = model.predict(x_test_scaled)
        validation_predictions = model.predict(x_val_scaled)
        df_predictions[str(i)] = np.stack(predictions, axis=1)[0].tolist()
        df_validation_predictions[str(i)] = np.stack(validation_predictions, axis=1)[0].tolist()

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    stacked_model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
    stacked_model.fit(df_predictions, y_test)

    # Save model into a binary file
    pkl.dump(stacked_model, open('./src/stacked_model.pkl', 'wb'))

    print(f"***** Stacked Model *****")
    stacked_predictions = stacked_model.predict(df_validation_predictions)

    rmse = np.sqrt(mean_squared_error(y_val, stacked_predictions))
    print('Root Mean Squared Error:', rmse)
    with open('model_evaluation.txt', 'a') as file:
        file.write(f"Stacked Model RMSE: {rmse}\n")


def main():
    x_train, x_temp, y_train, y_temp = prepare_model_data()
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5)
    models = build_stand_alone_models(x_train, x_val, y_train, y_val)
    build_stacked_model(models, x_test, x_val, y_test, y_val)


if __name__ == '__main__':
    main()
