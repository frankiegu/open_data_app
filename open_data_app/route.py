from flask import Flask,render_template,jsonify,request
import pickle
import pdb

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict/<model_name>',methods=['GET','POST'])
def predict(model_name):
    # pdb.set_trace()
    print(model_name)
    data = request.json['data']
    sample = [float(x) for x in data]
    with open('open_data_app/static/model/iris.pkl','rb') as f:
        clf = pickle.load(f)
    result = clf.predict([sample])
    print(result)
    return jsonify(result.tolist())

