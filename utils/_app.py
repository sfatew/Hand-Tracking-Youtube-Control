from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from inference import GestureModel
from utils.processing import output_process
from utils.mediapipe_utils import init_holistic
import MSSTNET
import datetime
import time


MSST_Layer = MSSTNET.MSST_Layer

import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model_paths = [
    "best_model/joint_stream.keras",
    "best_model/joint_motion_stream.keras",
    "best_model/bone_stream.keras",
    "best_model/bone_motion_stream.keras"
]


try:
    model = GestureModel(model_paths, custom_objects={'MSST_Layer': MSST_Layer})
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    exit(1)
holistic = init_holistic()
frames = []

@socketio.on("frame")
def handle_frame(data):
    print("Received a frame")

    global frames
    # Decode base64 image string
    image_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    frames.append(frame)
    print(f"Buffer size: {len(frames)}")  # Add this

    if len(frames) >= 37:
        print("Processing 37 frames...")  # Add this
        try:
            start_time = time.time()
            sequences = output_process(frames, holistic)
            label, confidence = model.predict(sequences)
            print(f"Prediction time: {time.time() - start_time:.2f} seconds")  # Add this
            print(f"Predicted label: {label}, confidence: {confidence}")  # Add this
            emit("prediction", {"label": label, "confidence": confidence})

            
            #testing
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"Processi ng 37-frame batch at {timestamp}...")

            for i, f in enumerate(frames[:37]):
                filename = f"testimages/frames_{timestamp}_frame_{i+1:02d}.jpg"
                cv2.imwrite(filename, f)
                print(f"Saved {filename}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            emit("prediction", {"label": "Error", "confidence": 0})
        frames = []  # Reset

@app.route('/')
def index():
    return jsonify({"message": "Server is running."})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)