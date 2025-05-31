# Hand-Tracking-Youtbe-Control

This project leverages computer vision techniques to track hand motion and control a computer interface in a similar manner to a touchpad.

## Overview

The goal of this project is to develop a system that allows for hand gesture recognition using DSCNet. The system tracks hand movements and translates them into computer commands, offering a hands-free way to interact with digital devices.

## Paper Reference

### A Dense-Sparse Complementary Network for Human Action Recognition based on RGB and Skeleton Modalities
(https://www.sciencedirect.com/science/article/abs/pii/S0957417423035637)

### Multi-scale Spatial–Temporal Convolutional Neural Network for Skeleton-based Action Recognition  
May 2023. *Pattern Analysis and Applications*<br>
DOI: [10.1007/s10044-023-01156-w](https://doi.org/10.1007/s10044-023-01156-w)

## Dataset

The dataset used for training the model is Jester, which includes various video data that helps in recognizing hand action.
 
You can download the dataset from Kaggle:  
[20bn-jester Dataset on Kaggle](https://www.kaggle.com/datasets/toxicmender/20bn-jester)

## 🚀 How to Run

### Prerequisites
- Python 3.9 to 3.12
- Webcam connected

Follow these steps to set up and run the **Hand-Tracking-Youtube-Control** project:

### 1. **Clone the repository and install the required packages**  
Open your terminal and run:
```bash
git clone https://github.com/sfatew/Hand-Tracking-Youtube-Control.git
cd Hand-Tracking-Computer-Control

# We recommend creating a virtual environment before installing dependencies.
python -m venv venv

# Activate the virtual environment:
venv\Scripts\activate # On Windows
source venv/bin/activate # On macOS/Linux

pip install -r requirements.txt
```

### 2. Download the pre-trained model

- Download the `best_model` folder from [this link](https://drive.google.com/file/d/12gNNiUO1jzPNQfqIZ_4mgXttaTYKpfp2/view?usp=sharing).
- Place the entire `best_model` folder inside the `Hand-Tracking-Computer-Control` directory.

### 3. Connect your webcam
Ensure your webcam is connected before running the program.

### 4. Run the program
Start the UI by running:

```bash
python ui.py
```


> ⏳ Note: It may take about 2 minutes for the UI to appear due to the model loading time. We're working on improving this delay.

## 🎮 Usage
After the UI launches, open YouTube in your browser. Control playback hands-free using your gestures — no keyboard or mouse needed!

These are the actions users can perform:

| Gesture                        | Function                        |
|-------------------------------|----------------------------------|
| Rolling Hand Backward         | Previous chapter                 |
| Rolling Hand Forward          | Next chapter                     |
| Shaking Hand                  | Stop detect                      |
| Sliding Two Fingers Down      | Decrease volume                  |
| Sliding Two Fingers Up        | Increase volume                  |
| Sliding Two Fingers Left      | Rewind 5 seconds                 |
| Sliding Two Fingers Right     | Fast forward 5 seconds           |
| Stop Sign                     | Pause / Play                     |
| Swiping Down                  | Toggle captions                  |
| Swiping Up                    | Next video                       |
| Swiping Left                  | Previous browser tab             |
| Swiping Right                 | Next browser tab                 |
| Thumb Down                    | Mute / Unmute                    |
| Thumb Up                      | Enter / Exit fullscreen          |
| Turning Hand Clockwise        | Increase playback speed          |
| Turning Hand Counterclockwise| Decrease playback speed          |

## ⚙️ Optional: Adjust Parameters
You can customize a few settings for a better experience (default values are usually good):

- Cooldown Time: Time between each prediction. Prevents duplicate gestures being detected too quickly.

- Time to Collect Frame: Time window in which your action is captured.

👉 Make sure your gesture happens after the cooldown and within this collection time.

## 👥 Contributors

Thanks to these awesome people who contributed:

[![Contributors](https://img.shields.io/github/contributors/sfatew/Hand-Tracking-Computer-Control.svg)](https://github.com/sfatew/Hand-Tracking-Computer-Control/graphs/contributors)

---

⭐ If you like this project, please **give it a star** on GitHub!  
[![GitHub stars](https://img.shields.io/github/stars/sfatew/Hand-Tracking-Computer-Control?style=social)](https://github.com/sfatew/Hand-Tracking-Computer-Control/stargazers)
