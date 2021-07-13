import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier

iris = pd.read_csv("data/iris_training.csv")
testset = pd.read_csv("data/iris_test.csv")
train = iris.sample(frac=0.8)
test = iris.drop(train.index)

features = train.columns.drop('variety')
target = 'variety'

pipeline_obj = Pipeline([
    ("mapping", DataFrameMapper([
    (["sepal_length", "sepal_width"], StandardScaler()) , 
    (["petal_length", "petal_width"], SimpleImputer())
    ])),
    ("rfc", RandomForestClassifier(n_estimators = 100))
])

pipeline_obj.fit(train[features], train[target])

# Predicting
test['pred'] = pipeline_obj.predict(test[features])
print(test[['variety', 'pred']])

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(test['variety'], test['pred'])
print(cm)
print(accuracy_score(test['variety'], test['pred']))

# Predicting external testset
testset['pred'] = pipeline_obj.predict(testset[features])

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix, accuracy_score
print('testset:')
print(confusion_matrix(testset['variety'], testset['pred']))
print(accuracy_score(testset['variety'], testset['pred']))

from nyoka import skl_to_pmml

skl_to_pmml(pipeline_obj, features, target, "rf_pmml.pmml")
