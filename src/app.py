from flask import Flask, jsonify, render_template, request

from src import predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def digit_prediction():
    if request.method == "POST":
        img = request.get_json()
        data = predict.predict_image(img)
        return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
