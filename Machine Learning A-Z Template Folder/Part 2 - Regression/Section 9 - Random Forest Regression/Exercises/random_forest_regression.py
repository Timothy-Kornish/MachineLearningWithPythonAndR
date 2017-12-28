import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = pd.read_csv('Position_Salaries.csv')
print('\n-------------------------------------------------------------------\n')
print("Head of dataset:\n",dataset.head())
print('\n-------------------------------------------------------------------\n')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print("x:\n", x)
print('\n-------------------------------------------------------------------\n')
print("y:\n",y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                  Creating Random Forest Regressor
#-------------------------------------------------------------------------------

regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(x, y)
pred = regressor.predict(6.5)
print("prediction of salary for 6.5 years of experience: ", pred)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                  Visualizing Random Forest Regression
#-------------------------------------------------------------------------------
x_grid = np.arange(min(x), max(x), 0.001)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'r')
plt.plot(x_grid, regressor.predict(x_grid), c = 'blue')
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
