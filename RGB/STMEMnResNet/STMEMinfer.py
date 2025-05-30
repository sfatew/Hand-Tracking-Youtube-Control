import os 
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)

import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
import torch.nn.functional as F

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from model_implement.TSM import create_model

# Define argument parser
parser = argparse.ArgumentParser(description="STMEM hand action Inference")
parser.add_argument("--vid_path", type=str, required=True, help="Path to input vid")
parser.add_argument("--model_checkpoint", type=str, default=r"..\model\STMEM_TSM_RestNet50.pth", help="Path to model checkpoint")


def load_model(model_path, device, parallel = False):

    model = create_model(device)
    state_dict = torch.load(model_path, map_location=device)

    if parallel:
        model.load_state_dict(state_dict)

    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    if model is not None:
        print(f"Model loaded from {model_path}")
        
    return model

def preprocess_vid(vid, frame_size=(128, 128), max_frames=36):
    '''Preprocess the input video for STMEM model inference.
    Args:
        vid (list of np.array): List of frames (each frame is an np.array).
        frame_size (tuple of int): (H, W) to which each frame will be resized.
        max_frames (int): Number of frames to use (truncate or pad if needed).
    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, T*C, H, W)
    '''

    processed = []
    for i in range(max_frames):
        if i < len(vid):
            frame = vid[i]
        else:
            frame = np.zeros_like(vid[0])  # Pad with black frames

        frame = cv2.resize(frame, (frame_size[1], frame_size[0]))  # (W, H)
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame)

    video = np.stack(processed, axis=0).transpose(0, 3, 1, 2)  # (T, C, H, W)
    video = video.reshape(-1, frame_size[0], frame_size[1])     # (T*C, H, W)
    video_tensor = torch.from_numpy(video).float().unsqueeze(0) # (1, T*C, H, W)

    return video_tensor


def classify(model, video_tensor, usage = "standalone"):
    '''Classify the input video using the STMEM model.'''
    with torch.no_grad():
        output = model(video_tensor)  # Raw logits
        probs = F.softmax(output, dim=1)
        if usage == "standalone":
            pred_class = probs.argmax(dim=1).item()
            confidence = probs.max(dim=1).values.item()
            return pred_class, confidence
        elif usage == "ensemble":
            pred_class = probs.argmax(dim=1).item()
            confidence = probs.max(dim=1).values.item()
            return pred_class, confidence, np.array(probs.squeeze(0).tolist())
        else:
            raise ValueError("Invalid usage type. Use 'standalone' or 'ensemble'.")

        
def handle_load_vid(vid_path):
    pass

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model_checkpoint = args.model_checkpoint

    vid_path = args.image_path
    vid = handle_load_vid(vid_path)

    model = load_model(model_checkpoint, device)

    # Preprocess input image
    video_tensor = preprocess_vid(vid, device)

    label = classify(model, video_tensor, device)
    print(f"Label: {label}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)