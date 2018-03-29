import sys
from flask import Flask
from flask import jsonify
from flask_restful import Api
from flask_restful import Resource
from flask_restful import reqparse


app = Flask(__name__)
api = Api(app)

string_cols = ['sample_uuid']
int_cols = ['user_age']

parser = reqparse.RequestParser()
for col in string_cols:
    parser.add_argument(col, type=str)
for col in int_cols:
    parser.add_argument(col, type=int)


class Predict(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.load_model()

    def get(self):
        arguments = parser.parse_args()
        result = {
            'sample_uuid': arguments['sample_uuid'],
            'probability': 0.5,
            'label': 1.0
        }
        return jsonify(**result)

    def load_model(self):
        return None


api.add_resource(Predict, '/api/v1/predict')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        host = sys.argv[1]
        port = int(sys.argv[2])
    else:
        host = None
        port = None

    app.run(host=host, port=port, debug=False)
