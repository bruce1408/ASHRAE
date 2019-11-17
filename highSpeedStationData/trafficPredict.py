import os
import numpy as np
import pandas as pd
import sklearn
pd.set_option('display.max.columns', None)
totalPath = "/home/bruce/bigVolumn/Datasets/traffic_data/removeStr.csv"
df = pd.read_csv(totalPath, header=0, names=['Devices', 'Lane', 'PercentageCar', 'AverageDistance', 'TimeOccupy', 'Traffic', 'Speed','Time'])
# print(df.info())
print(df.describe())
print(df.head(15))