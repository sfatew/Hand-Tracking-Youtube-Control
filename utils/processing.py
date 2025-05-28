import numpy as np
import cv2
from .mediapipe_utils import mediapipe_detection, extract_keypoint
from .connections import POSE_CONNECTIONS_FILE_INDEX, LEFT_HAND_CONNECTIONS_FILE_INDEX, RIGHT_HAND_CONNECTIONS_FILE_INDEX


# Need output shape: (1, 37, 48, 3)
def output_process_msstnet(frames, model):
    SEQUENCE_LENGTH = 37
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



def crop_one_pic(keypoint_array_one_pic): # input: an array with 48 array inside
    pic_shape = (176, 100, 3)
    
    pose = keypoint_array_one_pic[:6]
    rh = keypoint_array_one_pic[6: 28]
    lh = keypoint_array_one_pic[28:]
    
    pose_scale = set()
    rh_scale = set()
    lh_scale = set()
    for point in pose:
        x = int((pic_shape[0] - 1) * min(point[0], 1))
        y = int((pic_shape[1] - 1) * min(point[1], 1))
        if x != 0 or y != 0:
            pose_scale.add((x, y))
    for point in rh:
        x = int((pic_shape[0] - 1) * min(point[0], 1))
        y = int((pic_shape[1] - 1) * min(point[1], 1))
        if x != 0 or y != 0:
            rh_scale.add((x, y))
    for point in lh:
        x = int((pic_shape[0] - 1) * min(point[0], 1))
        y = int((pic_shape[1] - 1) * min(point[1], 1))
        if x != 0 or y != 0:
            lh_scale.add((x, y))

    # SET the KEYPOINT
    key_point_set = pose_scale.union(rh_scale, lh_scale)
    
    x_set = [point[1] for point in key_point_set ]
    y_set = [point[0] for point in key_point_set ]
    # Sort the border for cropped picture
    if len(key_point_set) > 0: 
        min_x = min(x_set) - 10
        max_x = max(x_set) + 10
        min_y = min(y_set) - 10
        max_y = max(y_set) + 10
        min_x = max(0, min_x)
        max_x = min(pic_shape[1], max_x)
        min_y = max(0, min_y)
        max_y = min(pic_shape[0], max_y)
        return [min_x, max_x, min_y, max_y]
    else:
        return -1

def Cropped(frame_sequence, sequence):
    cropped_frame_sequence = frame_sequence.copy()
    
    crop_minX = crop_minY = 100000
    crop_maxX = crop_maxY = 0
    
    for i in range(len(sequence)):
        if type(crop_one_pic(sequence[i])) != int:
            ls = crop_one_pic(sequence[i])
            min_x, max_x, min_y, max_y = ls[0], ls[1], ls[2], ls[3]
            crop_minX = min(crop_minX, min_x)
            crop_minY = min(crop_minY, min_y)
            crop_maxX = max(crop_maxX, max_x)
            crop_maxY = max(crop_maxY, max_y)
        else:
            continue

    if crop_minX == 100000 and crop_minY == 100000 and crop_maxX == 0 and crop_maxY == 0 :
        print("RECORD AGAIN !!")
        
    for i in range(len(sequence)):
        pic = frame_sequence[i]
        cropped_frame_sequence[i] = pic[crop_minX:crop_maxX, crop_minY:crop_maxY]

    return cropped_frame_sequence

def output_process_resnet(frames, model):
    frames = [cv2.resize(frame, (176, 100)) for frame in frames]
    SEQUENCE_LENGTH = 32
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
    
    cropped_frame_sequence = Cropped(frames, sequence)
    cropped_frame_sequence = np.array(cropped_frame_sequence)
    resized_sequence = np.array([cv2.resize(frame, (128, 128)) for frame in cropped_frame_sequence])
    normalize_sequence = resized_sequence.astype(np.float32) / 255.0
    
    return np.expand_dims(normalize_sequence, axis=0)  # Shape: (1, sequence_length, 128, 128, 3)


def output_process_shiftgcn(frames, model):
    SEQUENCE_LENGTH = 37
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
    
    return np.expand_dims(sequence, axis=0)  # Shape: (1, sequence_length, 48, 3)