import pathlib
import pickle

import flask

import api


app = flask.Flask(__name__)

model_pkl = pathlib.Path.cwd() / 'data' / 'models' / 'rand_forest.pkl'
with open(model_pkl, 'rb') as fp:
    model = pickle.load(fp)


@app.route('/') # , methods=["POST", "GET"]
def index():
    return flask.render_template('essay.html')


@app.route('/', methods=['POST'])
def my_form_post():
    essay = flask.request.form['essay']
    prompt = flask.request.form['prompt']
    grade_level = flask.request.form['grade_level']
    feats = api.all_features(essay, prompt, grade_level)
    pred = model.predict(feats)
    return flask.render_template('scored.html')


if __name__ == '__main__':
    app.run(debug=True)
