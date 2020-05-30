import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing


path = "loan_train.csv" # load training data
df = pd.read_csv(path) # store into a dataframe

# print(df.head()) # to check if we've loaded the right data set
# print(df.shape)

#Convert due_date and effective_date attributes to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])


# Data visualisation
# print(df['loan_status'].value_counts()) #  tells us how many  paid off the loan and how many went into collection

# To create the bar charts by Principal and age on horizontal axis
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
# plt.show() # displays all bar charts created up to this point


#Pre-processing
# To see the day of the week people get loans
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
# plt.show()
# TREND those who get the loan at/near the end of the week don't tend to pay it off (blue dominates near end)

# Apply FEATURE BINARISATION to threshold this numerical data (day of week loan received) into boolean
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

# Does proportion of those who pay off vary with gender?
#print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))
# this shows 86% of women & 73% of men pay off
# Convert gender from boolean to numeric: 0 for male, 1 for female
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

#Does proportion of those who pay off vary with education attained?
#print(df.groupby(['education'])['loan_status'].value_counts(normalize=True))
# significant reduction in proportion only at 'Master or above'

# Apply ONE HOT ENCODING to convert categroical data (education) into a set of binary values that go on and off for each category
Feature = df[['Principal','terms','age','Gender','weekend']] # create a 'sub dataframe' with only the attributes of interest
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
#print(Feature.head()) # check if we've appended these binary variables to the sub dataframe 'Feature'
