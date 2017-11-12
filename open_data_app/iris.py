# iris建模
# linqingbin
# 20171111
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from collections import OrderedDict
# from open_data_app import modeling
import modeling
import numpy as np

def main():
    iris = datasets.load_iris()
    colnames = ["sepal length","sepal width","petal length",
        "petal width","class"]

    baseinfo = {
        "name":"iris",
        "name_cn":"鸢尾花品种预测",
        "sample_name":"花样"
    }

    features = OrderedDict({
        "sepal length":{
            "process":None,
            "describe":"萼片长度(cm)",
            'type':'numerical',
            },
        "sepal width":{
            "process":None,
            "describe":"萼片宽度(cm)",
            'type':'numerical',
            },
        "petal length":{
            "process":None,
            "describe":"花瓣长度(cm)",
            'type':'numerical',
            },
        "petal width":{
            "process":None,
            "describe":"花瓣宽度(cm)",
            'type':'numerical',
            },
    })
    target = {"class":{
        'process':None,
        'describe':'花种'}
        }
    Xy = np.vstack([iris.data.T,iris.target.T]).T

    data = {i:dict(zip(colnames,x)) for i,x in enumerate(Xy)}
    
    save_fp = 'open_data_app/static/model/iris.pkl'
    clf = GaussianNB()
    clf = modeling.modeling(clf,data,features,target,baseinfo,save_fp)

if __name__ == '__main__':
    main()

