import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error

#Load Data from CSV File
dataframe = pd.read_csv("titanic.csv")
dataframe = dataframe.drop(["Name"], axis=1)
print(dataframe.describe())

#Plot the Data
plt.title("Titanic Survival Figures")
plt.xlabel("")
plt.ylabel("")

ages = dataframe["Age"].values
fares = dataframe["Fare"].values
survived = dataframe["Survived"].values

colors = []
for item in survived:
    if(item == 0):
        colors.append('red')
    else:
        colors.append('green')

plt.scatter(ages, fares, s=50, color=colors)
plt.show()

#Build a NB Model
Features = dataframe.drop(['Survived'], axis=1).values
Targets = dataframe['Survived'].values

Features_train, Targets_train = Features[0:710], Targets[0:710]
Features_test, Targets_test = Features[710:], Targets[710:]

model = GaussianNB()

model.fit(Features_train, Targets_train)

predicted_values = model.predict(Features_test)
for x in zip(Targets_test, predicted_values):
    print(x)

#estimate the error
print("Accuracy is ", model.score(Features_test, Targets_test))