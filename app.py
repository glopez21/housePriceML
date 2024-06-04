from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    bedroom = request.form['bedrooms']
    bathroom = request.form['bathrooms']
    living = request.form['living']
    floors = request.form['floors']

    array = np.array([bedroom, bathroom, living, floors])
    array = array.astype(np.float64)

    pred = model.predict([array])

    return render_template('index.html', data=int(pred))


if __name__ == '__main__':
    app.run(debug=True)
