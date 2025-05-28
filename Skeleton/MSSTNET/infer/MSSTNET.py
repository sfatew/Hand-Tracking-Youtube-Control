import sys
import argparse
import os
import numpy as np
import mediapipe as mp
from tensorflow import keras
from pathlib import Path
from collections import OrderedDict

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Add the parent directory to sys.path
from model_implement import MSSTNET

def cal_sequence(sequence):
    """
    Input: <List> of keypoints from 37 frames of video -> Shape (37, 48, 3)
    """
    POSE_CONNECTIONS = frozenset([(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)])
    
    # Mapping from pose landmark IDs to file landmark IDs
    poseid2fileid = {12: 0, 14: 1, 16: 2, 11: 3, 13: 4, 15: 5}

    # Create file connections based on mapped pose IDs
    POSE_CONNECTIONS_FILE_INDEX = [
        (poseid2fileid[a], poseid2fileid[b])
        for (a, b) in POSE_CONNECTIONS
    ]

    "Connection" 
    HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

    HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

    HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

    HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

    HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

    HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

    HAND_CONNECTIONS = frozenset().union(*[
        HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
        HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
        HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
    ])

    handid2fileid = {a: a + 6 for a in range(21)}

    #Mapping from joint_id to index_id of array

    LEFT_HAND_CONNECTIONS_FILE_INDEX = [
        (handid2fileid[a], handid2fileid[b])
        for (a, b) in HAND_CONNECTIONS
    ]

    RIGHT_HAND_CONNECTIONS_FILE_INDEX = [
        (handid2fileid[a] + 21, handid2fileid[b] + 21)
        for (a, b) in HAND_CONNECTIONS
    ]

    #####################################
    sequence1 = np.expand_dims(sequence, axis = 0) # Shape: (1, sequence_length, 48, 3)
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

    return sequence1, np.expand_dims(sequence2, axis = 0), np.expand_dims(sequence3, axis = 0), np.expand_dims(sequence4, axis = 0)

def output(model_path1, model_path2, model_path3, model_path4, sequence):
    """
    Return the probabilities (one-hot-encoder form): 
    [prob of action1, prob of action2, prob of action3, 
    prob of action4, ..., prob of action18]
    - model_path1: joint stream
    - model_path2: joint motion stream
    - model_path3: bone stream
    - model_path4: bone motion stream
    """
    MSST_Layer = MSSTNET.MSST_Layer

    model1 = keras.load_model(model_path1, 
                       custom_objects={'MSST_Layer': MSST_Layer})
    model2 = keras.load_model(model_path2, 
                       custom_objects={'MSST_Layer': MSST_Layer})
    model3 = keras.load_model(model_path3, 
                       custom_objects={'MSST_Layer': MSST_Layer})
    model4 = keras.load_model(model_path4, 
                       custom_objects={'MSST_Layer': MSST_Layer})
    
    sequence1, sequence2, sequence3, sequence4 = cal_sequence(sequence)
    
    y1 = model1.predict(sequence1)
    y2 = model2.predict(sequence2)
    y3 = model3.predict(sequence3)
    y4 = model4.predict(sequence4)

    y = 1/4 * (y1 + y2 + y3 + y4)

    return y

def main(model_path1, model_path2, model_path3, model_path4, sequence):
    """
    Return the Action (String)
    """
    actions = ["Doing other things", "No gesture", 'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand', 
           'Sliding Two Fingers Down', 'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 'Sliding Two Fingers Up',
            'Stop Sign', 'Swiping Down','Swiping Left', 'Swiping Right', 'Swiping Up',
            'Thumb Down', 'Thumb Up',
            'Turning Hand Clockwise', 'Turning Hand Counterclockwise'
            ]
    label = actions[np.argmax(output(model_path1, model_path2, model_path3, model_path4, sequence))]
    return label
