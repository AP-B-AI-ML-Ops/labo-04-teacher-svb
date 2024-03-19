from flask import Flask, request, jsonify
import mlflow

RUN_ID = "18beee997acc489cb803dac4c063de7e"
logged_model = f'mlflow-artifacts:/1/{RUN_ID}/artifacts/model/MLmodel'
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s %s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(ride):
    features = prepare_features(ride)
    prediction = model.predict(features)
    return prediction[0]

# init flask application
app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_entrypoint():
    ride = request.get_json()
    pred = predict(ride)
    result = {
        "duration": pred,
        "model_version": RUN_ID
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)


# ride = {
#     "PULocationID": 10,
#     "DOLocationID": 50,
#     "trip_distance": 40
# }
# print(predict(ride))