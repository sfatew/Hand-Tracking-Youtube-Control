import cv2
import numpy as np
import time
import dearpygui.dearpygui as dpg
from collections import deque
from inference import MSSTNETModel, ResNetModel, STMEMnResNetModel, ShiftGCNModel, DSCNetEnsembleModel
from utils.controller import ComputerController


models = {
    "MSSTNET": MSSTNETModel(),
    "3DResNet": ResNetModel(),
    "STMEM + TSM-ResNet": STMEMnResNetModel(),
    "DSCNet (ensemble)": DSCNetEnsembleModel(),
    "ShiftGCN": ShiftGCNModel()
}
model_list = list(models.keys())

class ProgramState:
    """A class to manage the state of the program."""
    def __init__(self, fps, skip_interval=1):
        self.model_name = model_list[0]  # Default model name
        self.fps = fps
        self.skip_interval = skip_interval

        self.time_to_collect_frames = 1.0 # time to collect frames
        self.cooldown_time = 1.0 # cooldown time before next prediction
        self.SEQUENCE_LENGTH = 37

        self.frame_counter = 0
        self.collect_start_time = 0.0  # time when frame collection started

        self.frame_buffer = deque(maxlen=self.SEQUENCE_LENGTH)  # buffer to hold frames
        self.pred_label = "Not predicted yet"
        self.confidence = 0.0
        self.is_predicting = False
        self.last_prediction_time = None

        self.status_text = "Waiting for frames..."
        self.controller = ComputerController()

        self.image_enhancement_enabled = False  # Flag for image enhancement

    def set_model(self):
        """Set the model name from the radio button selection."""
        self.model_name = dpg.get_value("model_selector")
        print(f"Selected model: {self.model_name}")
        if self.model_name == "MSSTNET":
            self.SEQUENCE_LENGTH = 37
        elif self.model_name == "3DResNet":
            self.SEQUENCE_LENGTH = 32
        elif self.model_name == "STMEM + TSM-ResNet":
            self.SEQUENCE_LENGTH = 36
        elif self.model_name == "ShiftGCN":
            self.SEQUENCE_LENGTH = 37
        elif self.model_name == "DSCNet (ensemble)":
            self.SEQUENCE_LENGTH = 37

        self.skip_interval = max(1, round(self.fps * self.time_to_collect_frames / self.SEQUENCE_LENGTH))
        print(f"Sequence length set to {self.SEQUENCE_LENGTH}, skip interval changes to {self.skip_interval} frames.")

    def set_values_from_sliders(self):
        """Set values from the sliders."""
        self.time_to_collect_frames = dpg.get_value("time_to_collect_frames_slider")
        self.cooldown_time = dpg.get_value("cooldown_time_slider")
        print(f"Time to collect frames: {self.time_to_collect_frames}, Cooldown time: {self.cooldown_time}")
        
        self.skip_interval = max(1, round(self.fps * self.time_to_collect_frames / self.SEQUENCE_LENGTH))
        print(f"Set values from sliders and updated skip interval: {self.skip_interval} frames.")
    
    def update_fps(self, cap, measure_interval = 1.5):
        """Estimate FPS once"""
        frame_count = 0
        start_time = time.time()
        while time.time() - start_time < measure_interval:  # Measure for measure_interval seconds
            ret, _ = cap.read()
            if not ret:
                continue
            frame_count += 1
        fps = frame_count / measure_interval
        skip_interval = max(1, round(fps * self.time_to_collect_frames / self.SEQUENCE_LENGTH))
        self.fps = fps
        self.skip_interval = skip_interval
        dpg.set_value("fps_text", f"FPS: {fps:.2f}, Skip Interval: {skip_interval}")
    
    def update_fps_continuously(self, cap):
        """Continuously update FPS (run in another thread)."""
        while True:
            self.update_fps(cap)
            time.sleep(0.5)
    
    def change_status_text(self, text):
        """Change the status text."""
        self.status_text = text
        dpg.set_value("status_text", self.status_text)
        if self.status_text.startswith("Collecting frames") or self.status_text.startswith("Cooldown") or self.status_text.startswith("Model is predicting"):
            return
        print(self.status_text)

    def change_last_gesture_text(self, text):
        """Change the last gesture text."""
        dpg.set_value("last_gesture_text", text)
        print(text)

    def add_frame(self, frame):
        """Add a frame to the buffer every skip_interval frames."""
        if frame is None:
            print("Error: Frame is None.")
            return
        self.frame_counter += 1
        if self.frame_counter % self.skip_interval == 0:
            if len(self.frame_buffer) == 0:
                self.collect_start_time = time.time()  # first frame
            self.frame_buffer.append(frame.copy())
            self.change_status_text(f"Collecting frames: {len(self.frame_buffer)}/{self.SEQUENCE_LENGTH}")

    def is_cooldown_complete(self):
        """Check if the cooldown period is complete."""
        if self.last_prediction_time is None:
            return True
        if time.time() - self.last_prediction_time >= self.cooldown_time:
            return True
        
        return False
    
    def is_ready_for_prediction(self):
        """Check if we have some frames collected, cooldown period complete, and not currently predicting."""
        is_target_duration_reached = (time.time() - self.collect_start_time) >= self.time_to_collect_frames
        return len(self.frame_buffer) > 0 and is_target_duration_reached and self.is_cooldown_complete() and not self.is_predicting
    
    def run_inference(self):
        """Run inference on the collected frames.
        The conditions for running this function are: enough frames collected, cooldown period complete, and not currently predicting.
        """        
        assert self.is_predicting == True, "run_inference should not be called when is_predicting is False"

        #try:
        frames = list(self.frame_buffer)
        start_prediction_time = time.time()
        label, conf = models[self.model_name].predict(frames)
        self.pred_label = label
        self.confidence = conf
        self.last_prediction_time = time.time()
        self.change_last_gesture_text(f"Last gesture: {self.pred_label} (Confidence: {self.confidence:.2f} by {self.model_name} in {self.last_prediction_time - start_prediction_time:.2f}s)")
        
        self.controller.perform_action(self.pred_label)
        # except Exception as e:
        #     self.change_last_gesture_text(f"Error during inference")
        #     self.last_prediction_time = time.time()
        #     print(f"Prediction error: {e}")
        
        self.is_predicting = False

    def toggle_image_enhancement(self):
        """Toggle image enhancement on or off."""
        self.image_enhancement_enabled = not self.image_enhancement_enabled
        dpg.set_value("image_enhancement_checkbox", self.image_enhancement_enabled)
        if self.image_enhancement_enabled:
            print("Image enhancement enabled.")
        else:
            print("Image enhancement disabled.")


def enhance_image(frame): 
    """Enhance the image by applying CLAHE and gamma correction.
    Input frame is expected to be in BGR format.
    The output will be in BGR format as well."""
    if frame is None:
        print("Error: Frame is None.")
        return None
    
    # CLAHE
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Gamma correction
    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
      
    def calculate_gamma_hsv_method(img):
        import math
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(hsv)

        mid = 0.5
        mean = np.mean(val)
        gamma = math.log(mid*255)/math.log(mean)
        return gamma

    gamma = calculate_gamma_hsv_method(frame)
    frame = adjust_gamma(frame, gamma)

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV compatibility


def create_texture_data(frame, scale_percent=60):
    """Create texture data from a given frame.
    Args:
        frame (numpy.ndarray): The input frame from the camera.
        scale_percent (int): The percentage to scale the frame size for display purposes.
    Returns:
        tuple: A tuple containing the width, height, and texture data.
    """
    if frame is None:
        print("Error: Frame is None.")
        return None

    # Or resize by scale
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)



    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    data = frame.ravel()
    data = np.asfarray(data, dtype='f')
    texture_data = np.true_divide(data, 255.0)

    return frame.shape[1], frame.shape[0], texture_data

hist_data = {
    'b': [0]*256,
    'g': [0]*256,
    'r': [0]*256
}

def update_histogram(frame):
    """Update the histogram data based on the current frame."""
    global hist_data
    if frame is None:
        print("Error: Frame is None.")
        return

    # Calculate histogram for each channel
    for i, col in enumerate(('b', 'g', 'r')):
            histr = cv2.calcHist([frame], [i], None, [256], [0, 256])
            hist_data[col] = histr.flatten()
    for col in ('b', 'g', 'r'):
            dpg.set_value(f"{col}_series", [list(range(256)), hist_data[col].tolist()])

