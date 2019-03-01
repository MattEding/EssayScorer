import flask

import api


app = flask.Flask(__name__)


@app.route('/', methods=["POST", "GET"]) #
def index():
    args = flask.request.args

    essay = args.get('essay', '')
    prompt = args.get('prompt', '')
    grade_level = args.get('grade_level')
    if grade_level is not None:
        grade_level = int(grade_level)

    score = ''
    if all(arg for arg in [essay, prompt, grade_level]):
        try:
            features = api.all_features(essay, prompt, grade_level)
            score = api.score_essay(features)
        except Exception:
            pass

    return flask.render_template('essay.html', essay=essay, prompt=prompt,
                                 grade_level=grade_level, score=score)


if __name__ == '__main__':
    app.run(debug=True)
