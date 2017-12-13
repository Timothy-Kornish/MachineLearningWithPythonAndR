import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


#-------------------------------------------------------------------------------
#                            Data Preprocessing
#-------------------------------------------------------------------------------

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # all rows, all columns except the last
y = dataset.iloc[:, 3].values # all rows, only last column
print('\n-------------------------------------------------------------------\n')
print(x)
print('\n-------------------------------------------------------------------\n')
print(y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                         Filling in Missing Data
#-------------------------------------------------------------------------------

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_x = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer_x.transform(x[:, 1:3])
print(x)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                        Encoding Categorical Data
#-------------------------------------------------------------------------------

encoder_x = LabelEncoder()
x[:, 0] = encoder_x.fit_transform(x[:, 0])
print(x)
print('\n-------------------------------------------------------------------\n')

hotEncoder_x = OneHotEncoder(categorical_features = [0])
x = hotEncoder_x.fit_transform(x).toarray()
print(x)
print('\n-------------------------------------------------------------------\n')

encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y)
print(y)
print('\n-------------------------------------------------------------------\n')

#-------------------------------------------------------------------------------
#                        train test split of data
#-------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#-------------------------------------------------------------------------------
#                        Feature Scaling
#-------------------------------------------------------------------------------

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
