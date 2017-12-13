import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = pd.read_csv('Salary_Data.csv')
print('\n-------------------------------------------------------------------\n')
print("Head of dataset:\n",dataset.head())
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
print('\n-------------------------------------------------------------------\n')
print("x:\n", x)
print('\n-------------------------------------------------------------------\n')
print("y:\n",y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                            train test split
#-------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#-------------------------------------------------------------------------------
#                   Fitting a Linear Regression Model
#-------------------------------------------------------------------------------

regressor = LinearRegression()
regressor.fit(x_train, y_train)
predict = regressor.predict(x_test)

#-------------------------------------------------------------------------------
#               Plotting Linear Regression Model with matplotlib
#-------------------------------------------------------------------------------

plt.scatter(x_train, y_train, c = 'red')
plt.plot(x_train, regressor.predict(x_train), c = 'blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test, y_test, c = 'r')
plt.plot(x_test, predict, c = 'b')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

#-------------------------------------------------------------------------------
#               Plotting Linear Regression Model with seaborn
#-------------------------------------------------------------------------------

sns.lmplot(data = dataset, x = 'YearsExperience', y = 'Salary')
plt.show()
