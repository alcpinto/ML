# Eclat Association Rules Learning

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Data Preprocessing
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + ('data/' if dir_path.endswith('/') else '/data/')
dataset = pd.read_csv(data_path + 'Market_Basket_Optimisation.csv', header = None)
# transform the dataframe into the format expected by the apriori function
basket_trans = []
for i in range(0, 7501):
    basket_trans.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training the Eclat model in the dataset
from apyori import apriori
# min_support -> we want at least products that were bought 3 or more times a week
# 3 * 7 (days) / 7051 = 0.003
rules = apriori(transactions = basket_trans, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_lenght = 2, max_length = 2)

# Visualising the resuls
## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print(results)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results non sorted
print('Displaying the results non sorted')
print(resultsinDataFrame)

## Displaying the results sorted by descending supports
print('Displaying the results sorted by descending supports')
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))
