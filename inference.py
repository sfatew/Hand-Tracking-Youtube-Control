import numpy as np
from tensorflow import keras
from utils.processing import output_process
from utils.mediapipe_utils import init_holistic
import model_implement.MSSTNET as MSSTNET

class GestureModel:
    def __init__(self, model_name='MSSTNET'):
        if model_name == 'MSSTNET':
            paths = [
                "best_model/joint_stream.keras",
                "best_model/joint_motion_stream.keras",
                "best_model/bone_stream.keras",
                "best_model/bone_motion_stream.keras"
            ]
            custom_objects = {'MSST_Layer': MSSTNET.MSST_Layer}
        else:
            model_paths = None
            custom_objects = None
        self.models = [
            keras.models.load_model(path, custom_objects=custom_objects)
            for path in paths
        ]
        self.actions = ["Doing other things", "No gesture", 'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand', 
           'Sliding Two Fingers Down', 'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 'Sliding Two Fingers Up',
            'Stop Sign', 'Swiping Down','Swiping Left', 'Swiping Right', 'Swiping Up',
            'Thumb Down', 'Thumb Up',
            'Turning Hand Clockwise', 'Turning Hand Counterclockwise'
            ]

    def predict(self, frames, model_name='MSSTNET'):
        if model_name == 'MSSTNET':
            holistic = init_holistic()
            sequences = output_process(frames, holistic)
            # Run predictions on all 4 streams
            outputs = [model.predict(seq) for model, seq in zip(self.models, sequences)]
            avg_output = sum(outputs) / len(outputs)
            final_index = np.argmax(avg_output)
            return self.actions[final_index], float(np.max(avg_output))
        else:
            raise ValueError("Unsupported model name. Please use 'MSSTNET'.")

class GestureModel_của_Minh:
    pass