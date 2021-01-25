from sklearn.datasets import *
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

clf = tree.DecisionTreeClassifier()
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

not_setosas = [ i!=list(iris['target_names']).index('setosa') for i in iris['target'] ]
iris['data'] = iris['data'][not_setosas]
iris['target'] = iris['target'][not_setosas]


clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf)

tree.plot_tree(clf);







fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn,
               #class_names=cn,
               filled = True);



clas = tree.DecisionTreeClassifier(max_depth=4)
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)



class TreeThingie:
    def __init__(self):
        self.n_outputs = 1

        self.n_features_ = 4
        self.n_features_in_ = 4
        self.n_classes_ = 2
        self.max_features_ = 4

class EmptyClass:

    def __init__(self):
        self.n_features_ = 4
        self.n_features_in_ = 4
        self.n_outputs_ = 1
        self.n_classes_ = 2
        self.max_features_ = 4
        self.classes_ = np.array([1, 2])
        self.tree_ = TreeThingie()
        self.criterion = 'myCriterion'


    def fit(self):
        pass


'''
tree_
'''




e = EmptyClass()

tree.plot_tree(e)
