import pandas as pd


def calc_nodes (neurons_i, neurons_o=1, samples=560, a=2):
    print(f"Optimal Neurons for {samples} samples:"
          f"{samples / (a * (neurons_i + neurons_o))}")


if __name__ == '__main__':
    df = pd.read_csv('./datasets/car_prices.csv')
    y = df.copy(True)
    y = y[['current price']]
    x = pd.get_dummies(df, columns=['condition', 'years', 'economy', 'rating'])
    x = x[['on road old', 'on road now', 'km', 'top speed', 'hp', 'torque', 'condition_1',
           'condition_2', 'condition_3', 'condition_4', 'condition_5', 'condition_6', 'condition_8', 'condition_9',
           'condition_10', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'economy_8', 'economy_10', 'economy_11',
           'economy_13', 'economy_14', 'economy_15', 'years_2', 'years_3', 'years_4', 'years_5', 'years_6']]
    calc_nodes(len(x.columns))
