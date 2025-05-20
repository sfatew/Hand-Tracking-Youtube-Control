import socketio
import cv2
import base64
import time

sio = socketio.Client()
sio.transport = 'websocket'
prediction_received = False  # flag to wait for server

@sio.on('prediction')
def on_prediction(data):
    global prediction_received
    print("Prediction received:", data)
    print(time.time() - start_time)
    prediction_received = True

sio.connect('http://localhost:5000')

cap = cv2.VideoCapture(0)
frames = []
print("Capturing 37 frames...")

start_time = time.time()

while len(frames) < 37:
    ret, frame = cap.read()
    if not ret:
        continue
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    sio.emit('frame', f"data:image/jpeg;base64,{jpg_as_text}")
    frames.append(frame)
    cv2.imshow('gfg', frame) 
    cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()

# Wait for prediction
print("Waiting for prediction...")
start = time.time()
while not prediction_received and time.time() - start < 20:
    time.sleep(0.1)

sio.disconnect()
