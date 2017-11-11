import pdb
import pickle

def test_structure():
    '''测试模型的结构是否可以随便变化'''
    with open('open_data_app/static/model/iris.pkl','rb') as f:
        result = pickle.load(f)
    pdb.set_trace()
    pre = result['model'].predict([[1,1,1,1]])
    print(pre)

