from flask import Flask, request, render_template
import numpy as np
import pickle

# load the model
model = pickle.load(open('models/model.pkl', 'rb'))

# initialize our app
app = Flask(__name__)

# home route
@app.route('/')
def home():
    return render_template('index.html')

# prediction route
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # converting numeric values to it's name
    flower_replacement = {
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    }
    prediction = [flower_replacement.get(x, x) for x in prediction]
    output =prediction[0]
    return render_template('index.html', prediction_text='{}'.format(output))

# Graph route
@app.route("/graph")
def index():                                                                                         
    return render_template("graph.html")


if __name__ == "__main__":
    app.run(debug=False)
