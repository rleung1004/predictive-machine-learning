import pandas as pd

FOLDER_PATH = "./datasets/"
CSV = "car_prices.csv"

df = pd.read_csv(FOLDER_PATH + CSV)

# Display all columns of the data frame.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

numeric_features = ['on road old', 'on road now', 'km', 'top speed', 'hp', 'torque']
categorical_features = ['years', 'rating', 'condition', 'economy']

df_numeric = df[numeric_features]
df_categorical = df[categorical_features]

print(df.head())
print(df.describe().T)
print("***** Numerical Features *****")
print(df_numeric.describe().T)
print("***** Categorical Features *****")
print(df_categorical.describe().T)
