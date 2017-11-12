# 美国成人建模
# linqingbin
# 20171111

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
import pdb
import modeling

def main():
    fp = 'open_data_app/static/adult.csv'
    df = pd.read_csv(fp)

    data = df.T.to_dict()
    
    save_fp = 'open_data_app/static/model/adult.pkl'
    # y = df['rich'].values.flatten()
    # mdf = df[['education','sex','age']]
    lb = preprocessing.LabelBinarizer()
    le = preprocessing.LabelEncoder()

    baseinfo = {
        "name":"adult",
        "name_cn":"美国人收入预测",
        "sample_name":"美国人"
    }

    features = OrderedDict({
        'sex':{
            'describe':"性别",
            'process':preprocessing.LabelBinarizer(),
            'type':'categorical',
            },
        'education':{
            'describe':"教育水平",
            'process':preprocessing.LabelBinarizer(),
            'type':'categorical',
        }
    })

    target = {
         'rich':{
            'describe':"年收入大于50K美元",
            'process':le
        }
    }

    # X = np.vstack([edu.T,sex.T]).T
    # y_transform = le.fit_transform(y)
    # 需要转换y为数值
    clf = DecisionTreeClassifier()
    clf = modeling.modeling(clf,data,features,target,baseinfo,save_fp)

if __name__ == '__main__':
    result = main()

