import json

import model
from flask import Flask, request
from json2html import *

app = Flask(__name__)

mod = model.KNNmodel()
i2n = model.isbn2name()


@app.route('/query')
def root():
    name = request.args.get('name')
    isbn = i2n.name2isbn(name)
    q_name = i2n.isbn2name(isbn)
    out_isbn, out_scores = mod(isbn, ret_k=5, ret_scores=True, fill_to_k=True)

    if out_isbn is None:
        return 'Error'

    ret_names = [i2n.isbn2name(isbn) for isbn in out_isbn]

    ret = json.dumps(
        {
            'query_name': q_name,
            'names': ret_names,
            'isbns': out_isbn,
            'scores': out_scores
        },
        sort_keys=True, indent=4
    )

    fmt = request.args.get('format')
    if fmt in ['true', 'True', '1']:
        ret = json2html.convert(ret, clubbing=False, encode=True)

    return ret


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
