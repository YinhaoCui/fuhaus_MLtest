from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pathlib
app = Flask(__name__)
model = load_model('fulhaus-test.h5')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    pred = model.predict(img)
    class_names = ['Bed','Chair','Sofa']
    result = {'class': class_names[pred.argmax()], 'probability': str(pred.max())}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)