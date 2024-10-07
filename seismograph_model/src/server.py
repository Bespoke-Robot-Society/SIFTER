from flask import Flask
from flask_cors import CORS
from .predict import predict

app = Flask(__name__)
CORS(app)


@app.get("/model-predictions")
def hello_world():
    predictions = predict()
    return "<p>Hello, World!</p>"
