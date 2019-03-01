import flask

import api


app = flask.Flask(__name__)


@app.route('/', methods=["POST", "GET"]) #
def index():
    args = flask.request.args
    essay, prompt, grade_level, score = api.process_args(args)
    return flask.render_template('essay.html', essay=essay, prompt=prompt,
                                 grade_level=grade_level, score=score)


if __name__ == '__main__':
    app.run(debug=True)
