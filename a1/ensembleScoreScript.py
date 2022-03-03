import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import load_model


MANUAL_FEATURE_LIST = ['on road old', 'on road now', 'km', 'condition_6', 'condition_7', 'condition_8', 'condition_9',
                       'condition_10', 'years_2', 'years_3', 'years_4', 'years_5', 'years_6', 'years_7']
SELECTED_FEATURE_LIST = ['on road old', 'on road now', 'km', 'top speed', 'hp', 'torque', 'condition_1', 'condition_2',
                         'condition_3', 'condition_4', 'condition_5', 'condition_6', 'condition_8', 'condition_9',
                         'condition_10', 'rating_1', 'rating_2', 'rating_4', 'rating_5', 'economy_8', 'economy_9',
                         'economy_12', 'economy_13', 'economy_14', 'economy_15', 'years_3', 'years_4', 'years_5',
                         'years_6', 'years_7']

df_train_columns = pd.DataFrame().append({
    'year_2': 0, 'year_3': 0, 'year_4': 0, 'year_5': 0, 'year_6': 0, 'year_7': 0,
    'rating_1': 0, 'rating_2': 0, 'rating_3': 0, 'rating_4': 0, 'rating_5': 0,
    'economy_8': 0, 'economy_9': 0, 'economy_12': 0, 'economy_13': 0, 'economy_14': 0, 'economy_15': 0,
    'condition_1': 0, 'condition_2': 0, 'condition_3': 0, 'condition_4': 0, 'condition_5': 0, 'condition_6': 0,
    'condition_7': 0, 'condition_8': 0, 'condition_9': 0, 'condition_10': 0,
}, ignore_index=True)


def prepare_model_data():
    df = pd.read_csv("datasets/car_prices_mystery.csv")
    df = pd.get_dummies(df, columns=['condition', 'rating', 'economy', 'years'])
    df = adjustDfColumns(df_train_columns, df)
    return df


def load_x_scaler(is_manual: bool):
    if is_manual:
        return pkl.load(open('./src/manual_feature_set_x_scaler.pkl', 'rb'))
    return pkl.load(open('./src/selected_feature_set_x_scaler.pkl', 'rb'))


def adjustDfColumns(dfTrainTestPrep, dfScore):
    trainTestColumns = list(dfTrainTestPrep.keys())
    scoreColumns = list(dfScore.keys())
    for i in range(0, len(trainTestColumns)):
        columnFound = False
        for j in range(0, len(scoreColumns)):
            if (trainTestColumns[i] == scoreColumns[j]):
                columnFound = True
                break
        # Add column and store zeros in every cell if
        # not found.
        if (not columnFound):
            colName = trainTestColumns[i]
            dfScore[colName] = 0
    return dfScore


def predict_stand_alone_models(models: list, x: pd.DataFrame) -> object:
    df_predictions = pd.DataFrame()
    for i, model_data in enumerate(models):
        model, is_manual = model_data
        x_scaler = load_x_scaler(is_manual)
        if is_manual:
            x_scaled = x_scaler.transform(x.copy(True)[MANUAL_FEATURE_LIST])
        else:
            x_scaled = x_scaler.transform(x.copy(True)[SELECTED_FEATURE_LIST])
        predictions = model.predict(x_scaled)
        df_predictions[str(i)] = np.stack(predictions, axis=1)[0].tolist()

    return df_predictions


def score():
    df_score = prepare_model_data()
    stacked_model = pkl.load(open("./src/stacked_model.pkl", 'rb'))

    stand_alone_models = [
        load_model('./src/ann_model_0.h5'),
        load_model('./src/ann_model_1.h5'),
    ]

    df_predictions = predict_stand_alone_models(stand_alone_models, df_score)
    stacked_predictions = stacked_model.predict(df_predictions)
    print(stacked_predictions)


if __name__ == "__main__":
    score()
