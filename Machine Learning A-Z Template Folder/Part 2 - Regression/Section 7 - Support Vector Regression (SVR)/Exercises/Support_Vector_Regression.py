import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = pd.read_csv('Position_Salaries.csv')
print('\n-------------------------------------------------------------------\n')
print("Head of dataset:\n",dataset.head())
print('\n-------------------------------------------------------------------\n')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
print("x:\n", x)
print('\n-------------------------------------------------------------------\n')
print("y:\n",y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                            Scaling Data
#-------------------------------------------------------------------------------

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#-------------------------------------------------------------------------------
#                      Support Vector Machine Regressor
#-------------------------------------------------------------------------------

regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)
predict = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print("prediction of salary:", predict)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                  Visualizing Support Vector Regression
#-------------------------------------------------------------------------------

plt.scatter(x, y, color = 'r')
plt.plot(x, regressor.predict(x), c = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
