# iris建模
# linqingbin
# 20171111
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
# from open_data_app import modeling
import modeling
import numpy as np

def main():
    iris = datasets.load_iris()
    colnames = ["sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "class"]
    Xy = np.vstack([iris.data.T,iris.target.T]).T

    data = {i:dict(zip(colnames,x)) for i,x in enumerate(Xy)}
    features = {x:{'process':None,'describe':None}  for x in ["sepal length", "sepal width", "petal length", "petal width",]}
    target = {x:{'process':None,'describe':None}  for x in ["class",]}
    save_fp = 'open_data_app/static/model/iris.pkl'
    clf = GaussianNB()
    clf = modeling.modeling(clf,data,features,target,save_fp)

if __name__ == '__main__':
    main()

