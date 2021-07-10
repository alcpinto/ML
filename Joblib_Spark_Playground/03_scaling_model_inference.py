import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType, ArrayType
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from pyspark.sql import SparkSession
from sklearn.utils import parallel_backend
from joblibspark import register_spark
from IPython.display import display

# Training dataset (single node)
X, y = make_classification(n_samples=1000, random_state=0)

param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
             "kernel": ['rbf', 'poly', 'sigmoid'],
             "shrinking": [True, False]}

grid_search = GridSearchCV(SVC(gamma='auto', random_state=0, probability=True),
                          param_grid=param_grid,
                          return_train_score=False,
                          cv=3,
                          n_jobs=-1)

register_spark() # register spark backend

with parallel_backend('spark',n_jobs=3):
   grid_search.fit(X, y)


# Inference of new data (spark cluster)
X_score, y_score = make_classification(n_samples=100, random_state=0)

best_model_from_gridsearch = grid_search.best_estimator_

n_features=20
column_names = [f'feature{i}' for i in range(n_features)]

pdf_X_score = pd.DataFrame(X_score, columns = column_names).reset_index().rename(columns = {'index': 'id'})

#Mocked Spark Dataframe with 100 mocked records for inferencing.
#In production this record could scale to millions
#and millions of records.
spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
df_X_score = spark.createDataFrame(pdf_X_score)

@F.udf(returnType=DoubleType())
def predict_udf(*cols):
  # cols will be a tuple of floats here.
  return float(best_model_from_gridsearch.predict((cols,)))

df_pred = df_X_score.select(
  F.col('id'),
  predict_udf(*column_names).alias('prediction')
)

display(df_pred)
