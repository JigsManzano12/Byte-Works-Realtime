# Real-time Violence Detection using realtime.py

## Overview

Realtime.py is a Python script that detects violence in video streams using SSD MobileNet V2 for person detection and a custom violence detection model. It records segments of detected violence into separate MP4 files and supports live streaming of processed frames.

## Features

- **Real-time Detection:** Detects violence in video streams using SSD MobileNet V2 for person detection and a custom violence detection model.
- **Video Recording:** Records segments of detected violence into separate MP4 files.
- **Live Streaming:** Converts processed frames into JPEG format for real-time streaming.

## Requirements

- Python 3.x
- OpenCV (cv2)
- TensorFlow
- Keras
- TensorFlow Object Detection API
- NumPy

## Setup

1. **Install Dependencies:**
   Make sure all dependencies are installed. You can use the following command to install the required packages:

   ```bash
   pip install opencv-python tensorflow keras numpy
   ```
Additionally, you may need to install the TensorFlow Object Detection API and its dependencies.

2. **Model Files:**
Download and place the SSD MobileNet V2 model in the specified directory (ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model).
Ensure the violence detection model is available at Model_Used/Violence_Model/Violence_model.h5.
Ensure the label map is available at models/research/object_detection/data/mscoco_label_map.pbtxt.
## Usage
1. **Prepare Video Files:**
- Place video files for testing in the test_video directory.
2. **Run realtime.py:**
-Execute the script using the following command:
```bash
python realtime.py
```
3. **Output:**
- Recorded violence segments will be saved in the static/recorded_fights directory.
- Real-time streaming of processed frames can be accessed (details depend on implementation).

## Details
## Functionality

- Person Detection:
Uses SSD MobileNet V2 to detect persons in each frame of the video stream.

- Violence Detection:
Uses a pre-trained violence detection model to classify sequences of frames as violent or non-violent.

- Video Recording:
Records segments of detected violence into MP4 files with a naming convention (fight_record_<index>.mp4).

- Live Streaming:
Converts processed frames to JPEG format for real-time streaming.

## Key Functions
- get_next_file_index(output_dir, file_prefix="fight_record_"): Determines the next available file index for saving video segments.
- save_video(frames, output_dir, frame_rate, file_index): Saves a sequence of frames as a video file.
- put_label_on_frame(frame, label, location, font_scale, font, color, thickness): Adds a label to a frame for display.
- detect_person_ssd(frame, ssd_model, category_index, person_tracker): Detects persons in a frame using SSD MobileNet V2 and tracks their presence.
- sharpen_frame(frame): Applies a sharpening filter to a frame.
- process_video_stream(): Main function to process video streams, detect violence, and handle recording and streaming.
##License
[Specify the license under which this script is distributed.]

## Author
Gerard Jose Manzano

## Notes
- Customize directories and file paths as needed for your setup.
- Adjust threshold values and parameters for sensitivity and performance tuning.
- Ensure videos in the test_video directory are in supported formats (e.g., mp4).
