from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
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
    data = request.json['data']
    fp = 'open_data_app/static/model/{}.pkl'.format(model_name)
    with open(fp,'rb') as f:
        model = pickle.load(f)
    # sample = [float(data[x]) for x in data]
    sample = []
    for x in data:
        process = model['features'][x]['process']
        value = np.array([data[x]])
        if process:
            value = process.transform(value)
        sample.append(value)

    sample = np.vstack([x.T for x in sample]).T

    
    # sample = [float(x) for x in sample]
    result = dict(
        prediction = model['clf'].predict(sample).tolist(),
        proba =  '{:.1%}'.format(model['clf'].predict_proba(sample).max())
        )
    # pdb.set_trace()
    return jsonify(result)

