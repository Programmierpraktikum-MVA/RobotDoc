import os
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from threading import Thread

app = Flask(__name__)
socketio = SocketIO(app)

timeout = 120

base_model = Resnet50_Arc_loss()

mycroft_hw = HotwordDetector(
    hotword="Photo",
    model = base_model,
    reference_file=os.path.join(samples_loc, "alexa_ref.json"),
    threshold=0.7,
    relaxation_time=2
)

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75,
)

def start_hotword_detection():

    #code aus alexa:
    mic_stream.start_stream()

    print("Say Photo ")
    last_wakeword_time = time.time()
    while True:
        frame = mic_stream.getFrame()
        result = mycroft_hw.scoreFrame(frame)
        if result == None:
            # no voice activity
            if time.time() - last_wakeword_time > timeout:
                break
            continue
        if (result["match"]):
            last_wakeword_time = time.time()
            print("Wakeword uttered", result["confidence"])
            # Event happened, so stop the camera
            socketio.emit('stop-webcam')

    print("No activity for", timeout, "seconds. Stopping script...")

@app.route('/')
def index():
    return render_template('index2.html')

@socketio.on('start-hotword-detection')
def handle_start_hotword_detection():
    Thread(target=start_hotword_detection).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)