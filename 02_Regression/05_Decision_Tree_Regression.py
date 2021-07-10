# Decision Tree Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + ('data/' if dir_path.endswith('/') else '/data/')
dataset = pd.read_csv(dir_path + 'Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression on the whole dataset
# No need from feature scaling
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
level = 6.5
y_pred = regressor.predict([[level]])

# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


