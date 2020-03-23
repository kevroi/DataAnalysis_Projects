import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

path = "5MCSEcatalog.csv"  # load data
df = pd.read_csv(path)  # store in dataframe


def clean(x, d):  # function to convert longitude and latitude into right form, 2nd parameter tells which dir is +ve
    df['i'] = np.where(x.str[-1] == d, 1, -1).astype(dtype=float,
                                                     copy=True)  # like doing an if-else statement on a column, astype stores as float
    x = x.str[:-1].astype(dtype=float, copy=True)
    x = x * df['i']
    return x


df['Lat.'] = clean(df['Lat.'], 'N')
df['Long.'] = clean(df['Long.'], 'E')

lm = LinearRegression()  # create linear regression object
# X = df['Long.']
# Y = df['Lat.']


# to first see how the data looks on a scatter plot
width = 10
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x='Long.', y='Lat.', data=df)  # this only works with seaborn 0.9
plt.show()
