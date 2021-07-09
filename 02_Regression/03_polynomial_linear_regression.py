# Polynomial Linear Regression

from matplotlib import colors
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

# Compare Linear Regression with Polynomial Regression
# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Traning the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
plt.scatter(X, y, color='red')
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
level = 6.5
y_pred_lin = lin_reg.predict([[level]])
print(y_pred_lin)

# Predicting a new result with Polynomial Regression
level_poly = poly_reg.transform([[level]])
y_pred_poly = lin_reg_2.predict(level_poly)
print(y_pred_poly)

