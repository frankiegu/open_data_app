from flask import Flask,render_template,jsonify,request
import pickle
import pdb

app = Flask(__name__)
app.config.from_object('open_data_app.config')

@app.route('/')
def index():
    names = ['iris','adult']
    models =[]
    for name in names:
        fp = 'open_data_app/static/model/{}.pkl'.format(name)
        with open(fp,'rb') as f:
            model = pickle.load(f)
        models.append(model)
    print(models)
    return render_template('index.html',models=models)

@app.route('/api/predict/<model_name>',methods=['GET','POST'])
def predict(model_name):
    # pdb.set_trace()
    print(model_name)
    data = request.json['data']
    sample = [float(x) for x in data]
    with open('open_data_app/static/model/iris.pkl','rb') as f:
        model = pickle.load(f)
    result = model['clf'].predict([sample])
    print(result)
    return jsonify(result.tolist())

