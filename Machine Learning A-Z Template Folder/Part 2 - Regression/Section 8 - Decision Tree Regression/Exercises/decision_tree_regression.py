import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
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
#                  Creating Decision Tree Regressor
#-------------------------------------------------------------------------------

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)
pred_65 = regressor.predict(6.5)
pred_62 = regressor.predict(6.2)
print(pred_65, pred_62)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                  Visualizing Decision Tree Regression
#-------------------------------------------------------------------------------
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'r')
plt.plot(x_grid, regressor.predict(x_grid), c = 'blue')
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
