import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, render_template
from flask import make_response
from flask_socketio import SocketIO, emit
import cv2
import base64
from face_regc import face_stream



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
async_mode = None
socketio = SocketIO(app)


face_regc = face_stream.FaceStream(
            input_dir = './face_regc/input_dir',
            modelfile = './face_regc/pre_model/20180402-114759/20180402-114759.pb',
            mtcnn_dir = './face_regc/d_npy',
            classifier_filename = './face_regc/my_class/my_classifier.pkl'
        )

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.on('frame')
def frame_message(message):

    print("received")
    fd = open("./face_regc/tmp/input.jpg", "wb")
    decoded = base64.b64decode(message['data'])
    fd.write(decoded)
    fd.close()


    image = face_regc.detect_faces("./face_regc/tmp/input.jpg")
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer)
    emit('frame', {'data': encoded_string })

    # with open("./face_stream/tmp/output.jpg", "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode('ascii')
    #     emit('frame', {'data': encoded_string })

@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)

@socketio.on('connect')
def test_connect():
    print('Client connected')
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':                             
    socketio.run(app, debug=True, certfile='cert.pem', keyfile='key.pem', port=5500)