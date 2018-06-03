import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
from face_regc import face_stream



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
async_mode = 'eventlet'
socketio = SocketIO(app, async_mode=async_mode)
training = False


face_regc = face_stream.FaceStream(
            input_dir = './face_regc/input_dir',
            modelfile = './face_regc/pre_model/20180402-114759/20180402-114759.pb',
            mtcnn_dir = './face_regc/d_npy',
            classifier_filename = './face_regc/my_class/my_classifier.pkl'
        )



import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.on('image_frame')
def frame_message(message):

    if training == True:
        return
        
    print("image_frame")
    fd = open("./face_regc/tmp/input.jpg", "wb")
    decoded = base64.b64decode(message['data'])
    fd.write(decoded)
    fd.close()


    image = face_regc.detect_faces("./face_regc/tmp/input.jpg", "./face_regc/tmp/output.jpg")
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer).decode('ascii')
    emit('image_frame', {'data': encoded_string })


@socketio.on('add_input')
def add_input(message):
    print("add_input")

    file = id_generator(12)
    directory = "./face_regc/input_dir/" + message['name']
    if not os.path.exists(directory):
        os.makedirs(directory)

    fd = open(directory + "/" + file+ ".jpg", "wb")
    decoded = base64.b64decode(message['data'])
    fd.write(decoded)
    fd.close()

@socketio.on('re_train')
def retrain():
    training = True
    os.system("cd ./face_regc && python3 ./aligndata_first.py && python3 ./create_classifier_se.py")
    face_regc.load_model()
    training = False


@socketio.on_error()
def chat_error_handler(e):
    print('An error has occurred: ' + str(e))


@socketio.on('connect')
def test_connect():
    print('Client connected')
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':                             
    socketio.run(app, debug=True, port=5100)