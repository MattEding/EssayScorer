import flask

import src.scorer.api


app = flask.Flask(__name__)


@app.route('/', methods=["POST", "GET"]) #
def index():
    args = flask.request.args
    essay, prompt, grade_level, score = src.scorer.api.process_args(args)
    return flask.render_template('essay.html', essay=essay, prompt=prompt,
                                 grade_level=grade_level, score=score)


if __name__ == '__main__':
    # needs use_reloader otherwise cannot import scorer.api
    app.run(use_reloader=False)
