import pandas as pd
from pandas_profiling import ProfileReport

FOLDER_PATH = "./datasets/"
CSV = "car_prices.csv"

df = pd.read_csv(FOLDER_PATH + CSV)

profile = ProfileReport(df, title="Car Prices EDA", explorative=True)
profile.to_file("eda.html")
