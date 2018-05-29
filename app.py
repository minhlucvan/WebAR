
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import base64
from face_regc import face_stream
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

face_regc = face_stream.FaceStream(
            input_dir = './face_regc/input_dir',
            modelfile = './face_regc/pre_model/20180402-114759/20180402-114759.pb',
            mtcnn_dir = './face_regc/d_npy',
            classifier_filename = './face_regc/face_stream/my_class/my_classifier.pkl'
        )


@app.route('/public/<path:path>')
def index(path):
    return send_from_directory('public', path)

@socketio.on('frame')
def on_frame(message):
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

@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app)
	