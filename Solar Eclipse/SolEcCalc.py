import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression


path = "5MCSEcatalog.csv" #load data
df = pd.read_csv(path) #store in dataframe
#print(df.head())

lm = LinearRegression() #create linear regression object
# X = df[] need to find a way to store juist the numeric value of lat in here, and use N?S to indicate positive or negative
