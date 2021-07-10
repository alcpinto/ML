from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X, y = make_classification(n_samples=1000, random_state=0)

param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
             "kernel": ['rbf', 'poly', 'sigmoid'],
             "shrinking": [True, False]}

grid_search = GridSearchCV(SVC(gamma='auto', random_state=0, probability=True),
                          param_grid=param_grid,
                          return_train_score=False,
                          cv=3,
                          n_jobs=-1)

grid_search.fit(X, y)