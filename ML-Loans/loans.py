import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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

# To normalise this one hot encoded Feature sub dataframe
X = Feature
X = preprocessing.StandardScaler().fit(X).transform(X) # standardise to mean of 0 and variance of 1
# print(X[0:5]) # to view the first 5 rows of standardised data


# K NEAREST NEIGHBOUR CLASSIFICATION
y = df['loan_status'].values # get our labels of attributes of interest
# to find the best K, split the training data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

Ks = 15 #   test 1 to 14 nearest neighbours
mean_acc = np.zeros(Ks-1) # mean accuracy
std_acc = np.zeros(Ks-1) # standard deviation of accuracy
for n in range(1, Ks):
    kNN_test_model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = kNN_test_model.predict(X_test)
    mean_acc[n-1] = np.mean(yhat==y_test)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
# print(mean_acc) # to see our mean accuracy for each value of k
k = np.argmax(mean_acc) + 1 # choose k to be the index of the max element in the array
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
# print(kNN_model) #check if it works
