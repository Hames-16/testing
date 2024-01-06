from flask import Flask, render_template, request
from flask_socketio import SocketIO
from process import process_image
import numpy as np
import json
import sys
import cv2


app = Flask(__name__)
socketio = SocketIO(app)


@app.get("/")
def hello_world():
    return render_template('index.html')


@app.post('/predict')
def predict():
    try:
        in_sample = request.form.get('symptom_sample')
        sample = None

        if in_sample is not None:
            symptom_sample = json.loads(in_sample)
            keys = sorted(list(symptom_sample.keys()))
            sample = np.zeros(len(keys))

            columns = ['Digestive discomfort', 'Dairy Allergies', 'Allergies', 'Nut allergies',
                       'Gas', 'Bloating', 'Blood sugar', 'Dental issues', 'Foodborne illnesses',
                       'cholesterol', 'Lactose intolerance', 'Gluten sensitivity/intolerance',
                       'Weight gain', 'Heart issues']

            for i, key in enumerate(columns):
                sample[i] = symptom_sample.get(key, None)

        image_storage = request.files['image']
        image_temp = image_storage.read()
        image_arr = np.fromstring(image_temp, np.uint8)
        image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        response = process_image(image_rgb, sample, socketio)

        return {
            'status': 'success',
            'data': response
        }

    except:
        return {
            'status': 'error',
            'data': None,
            'stacktrace': repr(sys.exception())
        }


if __name__ == '__main__':
    socketio.run(app)
