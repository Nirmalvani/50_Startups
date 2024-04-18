from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('Lr.pkl','rb'))
scaler= pickle.load(open("scaler.pkl","rb"))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/f', methods=['GET','POST'])
def predict():
    
    if request.method=='POST': 
        rd = float(request.form['rd'])
        ad = float(request.form['ad'])
        State = request.form['State']
        market = float(request.form['market'])

        if State == 'New York':
            state_value = 2
        elif State == 'California':
            state_value = 0
        else :
            state_value = 1

        scaled_values=scaler.transform([[rd,ad,market]])
        
        input_data = np.concatenate([scaled_values.flatten(), np.array([state_value])], axis=0)
        input_data = input_data.reshape(1, -1).astype(float)
        
        prediction=model.predict(input_data)
        output=round(prediction[0],2)

        return render_template('index.html', Pred=output)
    return render_template('index.html', Pred=None)


if __name__=="__main__":
    app.run(debug=True)