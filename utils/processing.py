import numpy as np
import cv2
from .mediapipe_utils import mediapipe_detection, extract_keypoint
from .connections import POSE_CONNECTIONS_FILE_INDEX, LEFT_HAND_CONNECTIONS_FILE_INDEX, RIGHT_HAND_CONNECTIONS_FILE_INDEX

SEQUENCE_LENGTH = 37

# Need output shape: (1, 37, 48, 3)
def output_process(frames, model):
    sequence = []
    
    # Process each frame to extract keypoints
    for frame_num in range(0, SEQUENCE_LENGTH):
        frame = frames[frame_num]
        if frame is None:
            print("IO Error")
            continue
        
        image, res = mediapipe_detection(frame, model)
        keypoints = extract_keypoint(res) 
        sequence.append(keypoints)  # Shape: (sequence_length, 48, 3)
    
    # 1. Joint stream
    sequence = np.array(sequence)
    sequence1 = np.expand_dims(sequence, axis = 0) # Shape: (1, sequence_length, 48, 3)

    # 2. Join motion stream
    sequence2 = sequence[1:] -sequence[:-1] # Shape: (sequence_length - 1, 48, 3)
    
    # 3. Bone stream
    sequence3 = []

    def retrieve_bone_vector(x):
        return np.stack([sequence[:, b, :] - sequence[:, a, :] for (a, b) in x], axis=1)
    
    bone_pose = retrieve_bone_vector(POSE_CONNECTIONS_FILE_INDEX)
    bone_left = retrieve_bone_vector(LEFT_HAND_CONNECTIONS_FILE_INDEX)
    bone_right = retrieve_bone_vector(RIGHT_HAND_CONNECTIONS_FILE_INDEX)    
    sequence3 = np.concatenate([bone_pose, bone_left, bone_right], axis=1) # Shape: (sequence_length, 47, 3)

    # 4. Bone motion stream
    sequence4 = sequence3[1:] - sequence3[:-1] # Shape: (sequence_length - 1, 47, 3)

    return np.expand_dims(sequence, axis = 0), np.expand_dims(sequence2, axis = 0), np.expand_dims(sequence3, axis = 0), np.expand_dims(sequence4, axis = 0)
