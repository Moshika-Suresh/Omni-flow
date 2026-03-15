from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    bus = request.form["bus"]
    stop = request.form["stop"]

    result = subprocess.run(
        ["python", "bus_prediction.py", bus, stop],
        capture_output=True,
        text=True
    )

    output = result.stdout

    return render_template("index.html", result=output)

if __name__ == "__main__":
    app.run(debug=True)
