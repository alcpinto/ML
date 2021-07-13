# Kernel SVM

import pandas as pd

iris_df = pd.read_csv("data/iris.csv")

iris_X = iris_df[iris_df.columns.difference(["variety"])]
iris_y = iris_df["variety"]

# le = LabelEncoder()
# iris_y = le.fit_transform(iris_y)

from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.preprocessing import LabelEncoder

pipeline = PMMLPipeline([
	("mapper", DataFrameMapper([
		(["sepal_length", "sepal_width", "petal_length", "petal_width"], [ContinuousDomain(), SimpleImputer()])
	])),
	("pca", PCA(n_components = 3)),
	("selector", SelectKBest(k = 2)),
	("classifier", LogisticRegression())
])
pipeline.fit(iris_X, iris_y)
pipeline.verify(iris_X.sample(n = 15))

pred = pipeline.predict(iris_X)


# from sklearn2pmml import sklearn2pmml

# sklearn2pmml(pipeline, "LogisticRegressionIris.pmml", with_repr = True)
