from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd


df = pd.read_csv("./datasets/car_prices.csv")
y = df[['current price']]
x = pd.get_dummies(df, columns=['condition', 'rating', 'economy', 'years'])
x = x.drop(['current price'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_best = SelectKBest(score_func=f_regression, k=30)
x_best.fit(x_train, y_train)

mask = x_best.get_support()
new_feat = []

for selected, feature in zip(mask, x_train.columns):
    if selected:
        new_feat.append(feature)
print('The best features are {}'.format(new_feat))
