import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# List of PoseLandmark we use in this project
from mediapipe.python.solutions.holistic import PoseLandmark
included_landmarks = [
    # right hand set
    PoseLandmark.RIGHT_SHOULDER, 
    PoseLandmark.RIGHT_ELBOW, 
    PoseLandmark.RIGHT_WRIST, 

    # left hand set
    PoseLandmark.LEFT_SHOULDER, 
    PoseLandmark.LEFT_ELBOW, 
    PoseLandmark.LEFT_WRIST, 
    ]

def init_holistic():
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert to RGB Color_space
    image.flags.writeable = False #cvt numpy to read-only
    res = model.process(image) #make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, res

def draw_styled_landmarks(image, res):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoint(res):
    arr = []

    # Pose landmarks (6 points)
    if res.pose_landmarks:
        for landmark_id in included_landmarks:
            point = res.pose_landmarks.landmark[landmark_id]
            arr.extend([point.x, point.y, point.z])
    else:
        arr.extend([0, 0, 0] * len(included_landmarks))  # 6*3 = 18

    # Left hand landmarks (21 points)
    if res.left_hand_landmarks:
        for point in res.left_hand_landmarks.landmark:
            arr.extend([point.x, point.y, point.z])
    else:
        arr.extend([0, 0, 0] * 21)

    # Right hand landmarks (21 points)
    if res.right_hand_landmarks:
        for point in res.right_hand_landmarks.landmark:
            arr.extend([point.x, point.y, point.z])
    else:
        arr.extend([0, 0, 0] * 21)

    return np.array(arr).reshape(48, 3)

