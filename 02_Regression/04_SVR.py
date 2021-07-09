# Support Vector Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import the dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + ('data/' if dir_path.endswith('/') else '/data/')
dataset = pd.read_csv(dir_path + 'Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
# Reshape y is required because StandardScaler expects a 2-dimensional array
from sklearn.preprocessing import StandardScaler
y = y.reshape((len(y), 1))
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Traning the SVR model on the whoe dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
level = 6.5
scaled_level = sc_X.transform([[6.5]])
y_pred = regressor.predict(scaled_level) 
y_inversed = sc_y.inverse_transform(y_pred)

# Visualising the SVR results
inverse_X = sc_X.inverse_transform(X)
inverse_y = sc_y.inverse_transform(y)
inverse_pred = sc_y.inverse_transform(regressor.predict(X))
plt.scatter(inverse_X, inverse_y, color = 'red')
plt.plot(inverse_X, inverse_pred, color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
plt.scatter(inverse_X, inverse_y, color = 'red')
X_grid = np.arange(min(inverse_X), max(inverse_X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
inverse_grid_y = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)))
plt.plot(X_grid, inverse_grid_y, color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
