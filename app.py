import sys
from flask import Flask
from flask import jsonify
from flask_restful import Api
from flask_restful import Resource
from flask_restful import reqparse
from sklearn.externals import joblib
import pandas as pd

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
        params = ['body_length', 'user_age']
        x = [[arguments.get(key) for key in params]]
        df = pd.DataFrame(data=x, columns=params).fillna(0)
        print(df)
        label = self.model.predict(df)[0]
        result = {
            'sample_uuid': arguments['sample_uuid'],
            'probability': 0.5,
            'label': label
        }
        return jsonify(**result)

    def load_model(self):
        lr = joblib.load('lr_model_y.pkl')
        return lr


api.add_resource(Predict, '/api/v1/predict')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        host = sys.argv[1]
        port = int(sys.argv[2])
    else:
        host = None
        port = None

    app.run(host=host, port=port, debug=False)
