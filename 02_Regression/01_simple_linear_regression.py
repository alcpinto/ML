# Simple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + ('data/' if dir_path.endswith('/') else '/data/') 
print(dir_path)
dataset = pd.read_csv(dir_path + 'Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Splitting the dataset into Traning set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Traning the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# Visualising the Training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction (employee with 12 years of experience)
print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.intercept_)
print(regressor.coef_)
# Salary = 26780.10 + 9312.58 * YearsExperience
