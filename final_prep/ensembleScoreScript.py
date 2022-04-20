import numpy as np
import pandas as pd
import pickle as pkl
from keras.models import load_model

PATH = "./datasets/car_prices_mystery.csv"

FEATURE_LIST = ['on road old', 'on road now', 'km', 'condition_6', 'condition_7', 'condition_8', 'condition_9',
                'condition_10', 'years_2', 'years_3', 'years_4', 'years_5', 'years_6', 'years_7']

df_train_columns = pd.DataFrame().append({
    'on road old': 0, 'on road now': 0, 'km': 0,
    'years_2': 0, 'years_3': 0, 'years_4': 0, 'years_5': 0, 'years_6': 0, 'years_7': 0,
    'condition_6': 0, 'condition_7': 0, 'condition_8': 0, 'condition_9': 0, 'condition_10': 0,
}, ignore_index=True)


def prepare_model_data():
    df = pd.read_csv(PATH)
    df = pd.get_dummies(df, columns=['condition', 'rating', 'economy', 'years'])
    df = adjustDfColumns(df_train_columns, df)
    return df


def load_x_scaler():
    return pkl.load(open('./src/x_scaler.pkl', 'rb'))


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


def predict_stand_alone_models(models: list, df: pd.DataFrame):
    df_predictions = pd.DataFrame()
    for i, model in enumerate(models):
        x_scaler = load_x_scaler()
        x_scaled = x_scaler.transform(df.copy(True)[FEATURE_LIST])
        predictions = model.predict(x_scaled)
        df_predictions[str(i)] = np.stack(predictions, axis=1)[0].tolist()

    return df_predictions


def score():
    df_score = prepare_model_data()
    stacked_model = pkl.load(open("./src/stacked_model.pkl", 'rb'))

    stand_alone_models = [
        load_model('./src/ann_model_0.h5'),
        load_model('./src/ann_model_1.h5'),
        load_model('./src/ann_model_2.h5')
    ]

    df_predictions = predict_stand_alone_models(stand_alone_models, df_score)
    stacked_predictions = stacked_model.predict(df_predictions)
    predictions = np.stack(stacked_predictions, axis=1)[0].tolist()
    df_stacked_predictions = pd.DataFrame({"current price": predictions})
    df_stacked_predictions.to_csv("car_prices_predictions.csv", index=False)


if __name__ == "__main__":
    score()
