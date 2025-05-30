import cv2
from collections import deque
import threading
import time
import datetime

# === Import your model and functions ===
from inference import GestureModel


model = GestureModel()

# === Constants ===
SEQUENCE_LENGTH = 37
TARGET_DURATION = 1.0
"""time to collect frames"""
PREDICTION_COOLDOWN = 1.0
"""time to wait before next prediction"""

# === Prediction state ===
system_state = {
    
}
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
pred_label = ""
confidence = 0.0
is_predicting = False
last_prediction_time = 0
fps, skip_interval = 0, 0

from utils.controller import ComputerController

# Initialize controller
controller = ComputerController()

# === Warmup to estimate FPS ===
def initial_estimate_fps(cap, warmup_time=2.0): # for the initial FPS estimation
    global fps, skip_interval
    print("⏳ Measuring webcam FPS...")
    frame_count = 0
    start_time = time.time()
    while time.time() - start_time < warmup_time:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    fps = frame_count / warmup_time
    skip_interval = max(1, round(fps * TARGET_DURATION / SEQUENCE_LENGTH))
    print(f"✅ Estimated FPS: {fps:.2f}")
    print(f"📸 Capturing every {skip_interval} frame(s) to get {SEQUENCE_LENGTH} frames over ~{TARGET_DURATION}s")


def update_fps(cap): # for the FPS estimation during the program (do in another thread)
    """Estimate FPS in a separate thread."""
    global fps, skip_interval
    while True:  # <-- Infinite loop
        frame_count = 0
        start_time = time.time()
        measure_interval = 1.0  # Measure for 2 seconds
        while time.time() - start_time < measure_interval:  # Measure 2 second
            ret, _ = cap.read()
            if not ret:
                continue
            frame_count += 1
        fps = frame_count / measure_interval
        skip_interval = max(1, round(fps * TARGET_DURATION / SEQUENCE_LENGTH))
        print(f'update fps: {fps:.2f}, skip_interval: {skip_interval}')
        #time.sleep(0.5)  # Sleep for a second before next measurement


# === Threaded prediction function ===
def run_inference(frames):
    global pred_label, confidence, is_predicting, last_prediction_time
    start_prediction_time = time.time()
    is_predicting = True
    try:
        # === Pad if too few frames ===
        if len(frames) < SEQUENCE_LENGTH:
            last_frame = frames[-1]
            needed = SEQUENCE_LENGTH - len(frames)
            frames.extend([last_frame] * needed)

        label, conf = model.predict(frames)
        pred_label = label
        confidence = conf
        print(f"Predicted label: {pred_label}, confidence: {confidence:.2f} from thread in {time.time() - start_prediction_time:.2f}s")
        controller.perform_action(pred_label)
        last_prediction_time = time.time()

    except Exception as e:
        pred_label = "Error"
        confidence = 0.0
        print("Prediction error:", e)
    is_predicting = False


if __name__ == "__main__":
    # === Start video capture ===
    cap = cv2.VideoCapture(0)    
    initial_estimate_fps(cap)
    threading.Thread(target=update_fps, args=(cap,), daemon=True).start()
    frame_counter = 0
    print("⏳ Starting video capture...")
    print("Press 'q' to exit.")
    while True: # collect -> predict -> cooldown -> reset buffer -> collect
        ret, frame = cap.read()
        if not ret:
            break

        #$ ----------------- collect frames for prediction -----------------
        # Flip the frame horizontally
        #frame = cv2.flip(frame, 1)
        frame_counter += 1
        current_time = time.time()

        # Track when we started collecting frames

        if frame_counter % skip_interval == 0:
            if len(frame_buffer) == 0:
                collect_start_time = current_time  # first frame
            frame_buffer.append(frame.copy())
            cv2.putText(frame, f"Collecting frames: {len(frame_buffer)}/{SEQUENCE_LENGTH}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)


        #$ ----------- predict -----------------
        # Trigger inference based on time passed (>= target duration) and cooldown
        if (not is_predicting and
            len(frame_buffer) > 0 and
            current_time - collect_start_time >= TARGET_DURATION and
            current_time - last_prediction_time > PREDICTION_COOLDOWN):
            
            frames = list(frame_buffer)
            threading.Thread(target=run_inference, args=(frames,), daemon=True).start()

        
        #$ ----------- cooldown -----------------
        if not is_predicting: # and pred_label:
            remaining = PREDICTION_COOLDOWN - (current_time - last_prediction_time)
            if remaining > 0: # is cooling down
                cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
                frame_buffer.clear() # clear buffer to avoid prediction
            

        # === Display status ===
        text = None
        if is_predicting:
            text = "Model is predicting..."
        elif pred_label:
            text = f"Last gesture: {pred_label} ({confidence:.2f})"

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        
        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()