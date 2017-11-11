# 美国成人建模
# linqingbin
# 20171111

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
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

    features = {
        'sex':{
            'describe':None,
            'process':lb,
            },
        'education':{
            'describe':None,
            'process':lb,
        }
    }

    target = {
         'rich':{
            'describe':None,
            'process':le
        }
    }

    # X = np.vstack([edu.T,sex.T]).T
    # y_transform = le.fit_transform(y)
    # 需要转换y为数值
    clf = DecisionTreeClassifier()
    clf = modeling.modeling(clf,data,features,target,save_fp)

if __name__ == '__main__':
    result = main()

