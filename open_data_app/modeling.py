# 建模通用组件
# linqingbin
# 20171111
import pickle
import numpy as np
import pdb
from sklearn.model_selection import cross_val_score

# def modeling(clf,X,y,save_fp):
#     scores = cross_val_score(clf,X,y, cv=5)
#     data = {'score':{
#         'mean':scores.mean(),
#         'std':scores.std()}
#      'model':clf.fit(X,y)
#     }
#     with open(save_fp,'wb') as f:
#         pickle.dump(data,f)
#     return data


def modeling(clf,data,features,target,save_fp):
    result = {
        'features':features,
        'target':target,}

    X = []
    
    for x in features:
        arr = np.array([data[i][x] for i in data])
        if result['features'][x]['process']:
            result['features'][x]['process'].fit(arr)
            transformed_values = result['features'][x]['process'].transform(arr)
            X.append(transformed_values)
        else:
            X.append(arr)

    X = np.vstack([x.T for x in X]).T

    for x in target:
        arr = np.array([data[i][x] for i in data])
        if result['target'][x]['process']:
            result['target'][x]['process'].fit(arr)
            transformed_values = result['target'][x]['process'].transform(arr)
            y = transformed_values
        else:
            y = arr
        break

    scores = cross_val_score(clf,X,y, cv=5)
    result.update({
        'score':{
            'mean':scores.mean(),
            'std':scores.std()
            },
        'model':clf.fit(X,y)
    }
    )
    with open(save_fp,'wb') as f:
        pickle.dump(result,f)
    return result
    

def save(clf,save_fp,score):
    data = {"model":clf,"score":score}
    with open(save_fp,'wb') as f:
        pickle.dump(data,f)

def read(save_fp):
    with open(save_fp,'rb') as f:
        model = pickle.load(f)
    return model

# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
# gnb.score(X_test,y_test)

# class Model():
#     def __init__(self,accuracy,two_std):
#         pass

