import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = pd.read_csv('Position_Salaries.csv')
print('\n-------------------------------------------------------------------\n')
print("Head of dataset:\n",dataset.head())
print('\n-------------------------------------------------------------------\n')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values
print("x:\n", x)
print('\n-------------------------------------------------------------------\n')
print("y:\n",y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                            Train Test Split
#-------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#-------------------------------------------------------------------------------
#                            Linear Regression
#-------------------------------------------------------------------------------

lin_regressor = LinearRegression()
lin_regressor.fit(x, y)

#-------------------------------------------------------------------------------
#                            Polynomial Regression with degree = 2
#-------------------------------------------------------------------------------

poly_regressor_2 = PolynomialFeatures(degree = 2)
x_poly_2 = poly_regressor_2.fit_transform(x)
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(x_poly_2, y)

#-------------------------------------------------------------------------------
#                            Polynomial Regression with degree = 3
#-------------------------------------------------------------------------------

poly_regressor_3 = PolynomialFeatures(degree = 3)
x_poly_3 = poly_regressor_3.fit_transform(x)
lin_regressor_3 = LinearRegression()
lin_regressor_3.fit(x_poly_3, y)

#-------------------------------------------------------------------------------
#                            Polynomial Regression with degree = 4
#-------------------------------------------------------------------------------

poly_regressor_4 = PolynomialFeatures(degree = 4)
x_poly_4 = poly_regressor_4.fit_transform(x)
lin_regressor_4 = LinearRegression()
lin_regressor_4.fit(x_poly_4, y)

#-------------------------------------------------------------------------------
#         Visualizing Linear Regression with varying degree of polynomial
#-------------------------------------------------------------------------------

plt.scatter(x, y, color = 'r')
plt.plot(x, lin_regressor.predict(x), c = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x, y, color = 'r')
plt.plot(x, lin_regressor_2.predict(poly_regressor_2.fit_transform(x)), c = 'blue')
plt.title("Truth or Bluff (Polynomial Regression degree = 3)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x, y, color = 'r')
plt.plot(x, lin_regressor_3.predict(poly_regressor_3.fit_transform(x)), c = 'blue')
plt.title("Truth or Bluff (Polynomial Regression degree = 3)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x, y, color = 'r')
plt.plot(x, lin_regressor_4.predict(poly_regressor_4.fit_transform(x)), c = 'blue')
plt.title("Truth or Bluff (Polynomial Regression degree = 4)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#-------------------------------------------------------------------------------
#  Visualizing smooth curve Linear Regression with varying degree of polynomial
#-------------------------------------------------------------------------------

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color = 'r')
plt.plot(x_grid, lin_regressor_4.predict(poly_regressor_4.fit_transform(x_grid)), c = 'blue')
plt.title("Truth or Bluff (Polynomial Regression degree = 4), smooth curve")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#-------------------------------------------------------------------------------
#                 Predicting a new result with Linear Regression
#-------------------------------------------------------------------------------

salary_pred = lin_regressor.predict(6.5)
print("Linear Regression Salary: ", salary_pred)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#               Predicting a new result with Polynomial Regression
#-------------------------------------------------------------------------------

salary_pred_poly_4 = lin_regressor_4.predict(poly_regressor_4.fit_transform(6.5))
print("Polynomial Regression Salary: ", salary_pred_poly_4)
print('\n-------------------------------------------------------------------\n')
