'''from sklearn import SVM
from sklearn import datasets
import numpy as np
from matpotlib import pyplot as plt
classifier = SVM.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
classifier.fit(X, y)'''
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# plotting cosmetics
plt.style.use('bmh')
# example of a linear model
from sklearn.linear_model import Perceptron
# example of a tree model
from sklearn.tree import DecisionTreeClassifier
def load_data():
    dat = pd.read_csv('./heart_disease.csv', header=None)
    data = dat.iloc[:, [0, 4]]
    target = dat.iloc[:, -1].replace([2, 3, 4], 1)
    return data.values, target.values
def make_data(n):
    np.random.seed(0)
    Y = np.random.choice([-1, +1], size=n)
    X = np.random.normal(size=(n, 2))
    for i in range(len(Y)):
        X[i] += Y[i]*np.array([-2, 0.9])
    return X, Y
x, y = load_data()
X, Y = make_data(100)
print(X)
print(x)
print(Y)
print(y)
plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='Paired_r', edgecolors='k')
plt.show()
