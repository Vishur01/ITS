from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the models

model3 = pickle.load(open('model3.pkl', 'rb'))
model4 = pickle.load(open('model4.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
            return redirect(url_for('input_page', model='model3_model4'))
        
    return render_template('index.html')


@app.route('/input_page/<model>', methods=['GET', 'POST'])
def input_page(model):
    pred3, pred4, tsr = None, None, None
    
    if request.method == 'POST':
        aggregate = request.form['Aggregate']
        source = request.form['source']
        viscosity = float(request.form['viscosity'])  # Assuming viscosity is a float
        dag = request.form['DAG']
        air_voids = request.form['air_voids']
        
        # Convert input data to a format that the model expects
        input_data = np.array([[aggregate, source, viscosity, dag, air_voids]])
        
        # Make prediction using the selected models
        if model == 'model3_model4':
            pred3 = np.round(model3.predict(input_data), 2)
            pred4 = np.round(model4.predict(input_data), 2)
            tsr = np.round(pred4 * 100 / pred3, 2) if pred3.all() != 0 else "Invalid prediction: Division by zero"

    return render_template('input_page.html', model=model, pred3=pred3, pred4=pred4, tsr=tsr)

if __name__ == '__main__':
    app.run(debug=True)
