import numpy as np
from tensorflow import keras

class GestureModel:
    def __init__(self, paths, custom_objects=None):
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

    def predict(self, sequences):
        # Run predictions on all 4 streams
        outputs = [model.predict(seq) for model, seq in zip(self.models, sequences)]
        avg_output = sum(outputs) / len(outputs)
        final_index = np.argmax(avg_output)
        return self.actions[final_index], float(np.max(avg_output))

class GestureModel_của_Minh:
    pass