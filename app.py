from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results')
def results():
    if not os.path.exists('results.json'):
        return jsonify({'status': 'not_ready',
                        'message': 'results.json not found. Run train.py first.'})
    with open('results.json') as f:
        data = json.load(f)
    return jsonify({'status': 'ok', 'results': data})


if __name__ == '__main__':
    app.run(debug=True)
