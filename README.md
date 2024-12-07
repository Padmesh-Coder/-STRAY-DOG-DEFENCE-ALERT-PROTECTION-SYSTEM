
# Stray Dog Deterent and Alert System

## Overview

This project implements a **real-time dog aggression detection system** using computer vision and deep learning. The system utilizes **YOLOv8** for object detection to identify aggressive dog behavior and triggers a deterrent using **ultrasonic vibrations** when aggression is detected. It also sends **SMS and WhatsApp alerts** using Twilio, and plays a sound to deter the aggressive behavior.

### Key Features:
- **Real-Time Detection**: Uses YOLOv8 for accurate, real-time dog behavior detection.
- **Alert System**: Sends SMS and WhatsApp notifications when aggressive behavior is detected.
- **Non-Invasive Deterrent**: Plays an audio sound as a deterrent for aggressive dogs.
- **Optimized for Performance**: Uses frame skipping and CUDA for fast processing on compatible systems.

## Table of Contents

- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Code](#running-the-code)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Results](#results)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rafael-ZP/Stray_Dog_Deterent_and_Alert_System.git
   cd dog-aggression-detection
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Prerequisites

- Python 3.x
- **YOLOv8** model for dog aggression detection (already trained models are provided)
- **dlib**: For accurate dog head tracking
- **Twilio account**: To send SMS and WhatsApp alerts (you will need to set up an account and obtain API keys).
- **OpenCV**: For video capture and real-time processing.
- **Playsound**: For playing deterrent sounds.
  
## Setup Instructions

1. **YOLOv8 Model**: The trained YOLO model (`best.pt`) and dog head detection model (`dogHeadDetector.dat`) are already stored in the `models` folder. Ensure that these files are present in the appropriate folder before running the script.

2. **Twilio Setup**: 
   - Sign up for a Twilio account at [Twilio](https://www.twilio.com/).
   - Obtain your **Account SID** and **Auth Token** from the Twilio dashboard.
   - Create a `.env` file in the root directory and add the following lines:

   ```
   AUTH_KEY=your_twilio_auth_token_1
   AUTH_KEY_2=your_twilio_auth_token_2
   ```

   Replace `your_twilio_auth_token_1` and `your_twilio_auth_token_2` with your actual Twilio API keys.

3. **Test video capture**: Make sure your webcam or video source is working properly, as the system requires video input to detect aggression.

## Running the Code

To run the code, simply execute the following command:

```bash
python main.py
```

### Expected Behavior:
- The system will start video capture from the default camera.
- **YOLOv8** will detect objects in real-time and classify them as aggressive or non-aggressive.
- If an aggressive dog is detected, the following actions will be triggered:
  - An **audio alert** (sound) will be played.
  - **SMS** and **WhatsApp alerts** will be sent to the configured numbers.
- Press `q` to exit the program.

## File Structure

The project directory is organized as follows:

```
dog-aggression-detection/
├── models/
│   ├── best.pt                 # Trained YOLOv8 model
│   └── dogHeadDetector.dat     # Dog head detection model (dlib)
├── .env                        # Twilio API keys
├── main.py                     # Main script for real-time detection
├── requirements.txt            # List of dependencies
└── README.md                   # This README file
```

## Configuration

### `data.yaml` Configuration:

The `data.yaml` file is used to configure the dataset paths for YOLOv8 training. Here's a sample configuration:

```yaml
path: '/content/gdrive/MyDrive/yolo/dataset'
train: /content/gdrive/MyDrive/yolo/dataset/train
val:  /content/gdrive/MyDrive/yolo/dataset/val

names:
  0: Aggressive
  1: Non_Aggressive
```

- The `train` and `val` paths point to the directories where the training and validation images are stored.
- The `names` section defines the classes (`Aggressive`, `Non_Aggressive`) that YOLOv8 will use for detection.

### Twilio Configuration:

In the `.env` file, you need to specify your Twilio authentication keys. Here's an example:

```
AUTH_KEY=your_twilio_auth_token_1
AUTH_KEY_2=your_twilio_auth_token_2
```

Ensure that these keys are valid to allow sending SMS and WhatsApp alerts.

## Results

The system provides **real-time feedback** with the following key outcomes:

1. **Detection**: When an aggressive dog is detected, the system draws a bounding box around the dog and labels it as `Aggressive`.
2. **Alerts**: Upon detecting aggression, SMS and WhatsApp alerts are sent to the configured phone numbers.
3. **Sound**: An alert sound is played as a deterrent.

### Screenshot Example:

![Detection Screenshot](![Picture1](https://github.com/user-attachments/assets/d1143839-3490-4c6e-a0da-d83e03bea569)


This screenshot shows the live detection window with bounding boxes around detected dogs, and labels indicating whether the dog is aggressive.

---

If you encounter any issues or need further assistance, feel free to open an issue on the GitHub repository. Enjoy using the Dog Aggression Detection System!
```
