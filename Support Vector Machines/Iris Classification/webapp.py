
from flask import Flask, render_template , request , jsonify
import pickle
import pandas as pd

app = Flask(__name__)

MODEL_PATH = r"./iris_SVC_model"

model = pickle.load(open(MODEL_PATH , "rb"))
target_names = ['setosa', 'versicolor', 'virginica']

@app.route("/predict", methods = ['POST'])
def predict() :
    jsonvalues = request.json
    x = pd.DataFrame(jsonvalues)
    y = model.predict(x)
    return jsonify({"Y " : [target_names[v] for v in y]})




if __name__ == "__main__" :

    app.run(debug= True)

