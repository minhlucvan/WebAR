
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import base64
from face_stream import face_stream

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/public/<path:path>')
def index(path):
    return send_from_directory('public', path)

@socketio.on('frame')
def test_message(message):
    print("received")

    fd = open("./face_stream/tmp/input.jpg", "wb")
    decoded = binary_data = base64.b64decode(message['data'])
    fd.write(decoded)
    fd.close()

    with open("./face_stream/tmp/output.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('ascii')
        emit('frame', {'data': encoded_string })

@socketio.on('my broadcast event')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True)

@socketio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    t = face_stream.FaceStream(1, print)
    t.start()
    socketio.run(app)
    t.cancel()
	