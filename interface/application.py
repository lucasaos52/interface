from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np

application = Flask(__name__, template_folder='./templates', static_folder='./static')

Pkl_Filename = "modelo.pkl" 
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)
@application.route('/')


def hello_world():
    return render_template('home.html')

@application.route('/predict', methods=['POST','GET'])
def predict():
    features = [int(x) for x in request.form.values()]

    df = pd.DataFrame({"matricula":[100],
                       "IDADE":[1],
                       "IMC":[1],
                       "FILHOS":[1],
                       "FUMANTE":[1],
                       "REGIÃO":[1],
                       "FACEBOOK":[1],
                       "CLASSE":[1],
                       "age_cat":[1],
                       "cat_peso":[1],
                       "SEXO_M":[1],
                      })

    print("df")
    print(df)
    
    print(features)
    final = np.array(features).reshape((1,6))
    print(final)
    pred = model.predict(final)[0]
    print(pred)

    
    if pred < 0:
        return render_template('op.html', pred='ERRO')
    else:
        return render_template('op.html', pred='ESTIMATIVA É: {0:.3f}'.format(pred))

if __name__ == '__main__':
    application.run(debug=True)