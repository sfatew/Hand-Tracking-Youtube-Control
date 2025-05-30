import numpy as np
from tensorflow import keras
from utils.processing import output_process_msstnet, output_process_resnet, output_process_shiftgcn
from utils.mediapipe_utils import init_holistic
import Skeleton.MSSTNET.model_implement.MSSTNET as MSSTNET
from RGB.STMEMnResNet.STMEMinfer import load_model, preprocess_vid, classify
import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder

class GestureModel:
    def __init__(self):
        self.holistic = init_holistic()
        self.actions = ["Doing other things", "No gesture", 'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand', 
           'Sliding Two Fingers Down', 'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 'Sliding Two Fingers Up',
            'Stop Sign', 'Swiping Down','Swiping Left', 'Swiping Right', 'Swiping Up',
            'Thumb Down', 'Thumb Up',
            'Turning Hand Clockwise', 'Turning Hand Counterclockwise'
            ]
    
    def padding_frames_naive(self, frames, target_length):
        """Pad frames to the target length."""
        if len(frames) < target_length:
            last_frame = frames[-1]
            needed = target_length - len(frames)
            frames.extend([last_frame] * needed)
        elif len(frames) > target_length:
            frames = frames[-target_length:]
        return frames
    
    def padding_frames(self, frames, target_length):
        """Pad frames to the target length by repeating frames cyclically to avoid bias."""
        current_length = len(frames)

        if current_length < target_length:
            # Repeat frames cyclically
            repeat_times = (target_length - current_length) // current_length
            remainder = (target_length - current_length) % current_length
            padded_frames = frames + frames * repeat_times + frames[:remainder]
            return padded_frames
        elif current_length > target_length:
            # Truncate from the beginning (keep most recent frames)
            return frames[-target_length:]
        else:
            return frames


class MSSTNETModel(GestureModel):
    def __init__(self):
        super().__init__()
        paths = [
            "best_model/joint_stream.keras",
            "best_model/joint_motion_stream.keras",
            "best_model/bone_stream.keras",
            "best_model/bone_motion_stream.keras"
        ]
        custom_objects = {'MSST_Layer': MSSTNET.MSST_Layer}

        self.models = [
            keras.models.load_model(path, custom_objects=custom_objects)
            for path in paths
        ]

    def predict(self, frames, usage="standalone"):
        frames = self.padding_frames(frames, 37)
        sequences = output_process_msstnet(frames, self.holistic)
        # Run predictions on all 4 streams
        outputs = [model.predict(seq) for model, seq in zip(self.models, sequences)] # shape: (4, 18)
        avg_output = sum(outputs) / len(outputs)
        final_index = np.argmax(avg_output)
        if usage == "ensemble":
            return self.actions[final_index], float(np.max(avg_output)), np.array(outputs)
        else:
            return self.actions[final_index], float(np.max(avg_output))

class ResNetModel(GestureModel):
    def __init__(self, model_path='best_model/3D_RestNet50(after48epoch).keras'):
        super().__init__()
        self.model = keras.models.load_model(model_path)
        
        self.label_encoder = OneHotEncoder(sparse_output=False)
        labels = np.array([[x] for x in self.actions])

        self.label_encoder.fit(labels)

    def predict(self, frames):
        frames = self.padding_frames(frames, 32)
        sequence = output_process_resnet(frames, self.holistic)
        output = self.model.predict(sequence)
        label = self.label_encoder.inverse_transform(output)
        return label[0][0], -1  # Confidence not available for ResNet model
    
class STMEMnResNetModel(GestureModel):
    def __init__(self, model_path='best_model/STMEM_TSM_RestNet50.pth', usage="standalone"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path=model_path, device=self.device, parallel=False)
        self.model.eval()
        self.model.to(self.device)
        self.usage = usage

    def predict(self, frames):
        #frames = self.padding_frames(frames, 37)
        video_tensor = preprocess_vid(frames)
        if self.usage == "standalone":
            final_index, confidence = classify(self.model, video_tensor)
            return self.actions[final_index], confidence
        elif self.usage == "ensemble":
            final_index, confidence, confidence_list = classify(self.model, video_tensor, usage=self.usage)
            return self.actions[final_index], confidence, confidence_list
    
class ShiftGCNModel(GestureModel):
    def __init__(self, model_path='best_model/best_model_shiftgcn.keras'):
        super().__init__()
        try:
            from Skeleton.ShiftGCN.ShiftGCN import initialize_shift_gcn_model
            self.model = initialize_shift_gcn_model()
            self.model.load_weights(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, frames):
        frames = self.padding_frames(frames, 37)
        sequences = output_process_shiftgcn(frames, self.holistic)
        output = self.model.predict(sequences)
        final_index = np.argmax(output)
        return self.actions[final_index], float(np.max(output))
    
class DSCNetEnsembleModel(GestureModel):
    def __init__(self, msst_f1_class_score_path='best_model/msst_f1_class_score.npy',
                 validation_metrics_path='best_model/validation_metrics.csv'):
        super().__init__()
        self.msstnet_model = MSSTNETModel()
        self.stmem_resnet_model = STMEMnResNetModel(usage="ensemble")
        self.msst_f1 = np.load(msst_f1_class_score_path)
        self.df = pd.read_csv(validation_metrics_path)

        array_str = self.df['f1_per_class'].iloc[0]
        self.rgb_f1 = np.fromstring(array_str.strip('[]"'), sep=' ')

    def predict(self, frames):
        dsc_score = np.zeros((18), dtype=float)
        _, _, msst_score_list_temp = self.msstnet_model.predict(frames, usage="ensemble")
        _, _, rgb_score = self.stmem_resnet_model.predict(frames)

        msst_score_list = msst_score_list_temp.reshape(4, 18)  # Reshape to (4, 18)

        for k in range(18):
            sum_f1 = 0.0
            for i in range(4):
                sum_f1 += self.msst_f1[i][k]
                dsc_score[k] += msst_score_list[i][k] * self.msst_f1[i][k]
            dsc_score[k] += rgb_score[k] * self.rgb_f1[k]
            sum_f1 += self.rgb_f1[k]
            dsc_score[k] /= sum_f1
        
        return self.actions[np.argmax(dsc_score)], float(np.max(dsc_score))