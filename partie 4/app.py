from flask import Flask, request, render_template
from predict import train_mlp, train_logistic_regression, get_mlp_predict, get_logistic_regression_predict

train_mlp()
train_logistic_regression()

app = Flask(__name__)

@app.route("/predict/", methods=["POST"])
def predict():
    avis = request.form["avis"]
    model = request.form["model"]
    
    if model == "mlp":
        return get_mlp_predict(avis)
    else:
        return get_logistic_regression_predict(avis)

@app.route("/")
def hello_world():
    return render_template("index.html")
