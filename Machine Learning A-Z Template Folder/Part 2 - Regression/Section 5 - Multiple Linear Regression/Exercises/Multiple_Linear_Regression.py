import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = pd.read_csv('50_Startups.csv')
print('\n-------------------------------------------------------------------\n')
print("Head of dataset:\n",dataset.head())
print('\n-------------------------------------------------------------------\n')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("x:\n", x)
print('\n-------------------------------------------------------------------\n')
print("y:\n",y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                            Encoding Data
#-------------------------------------------------------------------------------

label_encoder = LabelEncoder()
hot_encoder = OneHotEncoder(categorical_features = [dataset['State'].nunique()])

x[:, 3] = label_encoder.fit_transform(x[:, 3])
x = hot_encoder.fit_transform(x).toarray()
print("x hot encoded for seperate states:\n", x)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                            Avoiding Dummy variable trap
#-------------------------------------------------------------------------------

x = x[:, 1:]

#-------------------------------------------------------------------------------
#                            Train Test Split
#-------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#-------------------------------------------------------------------------------
#                      Fitting a Linear Regression Model
#-------------------------------------------------------------------------------

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#-------------------------------------------------------------------------------
#                      Predicting the test set results
#-------------------------------------------------------------------------------

prediction = regressor.predict(x_test)
comparison = np.array([abs(y_test - prediction)])
print("difference between prediction and actual values:\n", comparison)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
# Building the optimal model using Backward Elimination on Multiple Linear Regression
#-------------------------------------------------------------------------------

x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
# x1 and x2 p-value is very large, removing it
#-------------------------------------------------------------------------------

x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
# x4  p-value is very large, removing it, x5 is right at the significange level
#-------------------------------------------------------------------------------

x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
print('\n-------------------------------------------------------------------\n')
