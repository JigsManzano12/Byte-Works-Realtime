import cv2, os, time, threading, re
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import load_model
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def get_next_file_index(output_dir, file_prefix="fight_record_"):
    existing_files = os.listdir(output_dir)
    max_index = 0
    for file in existing_files:
        match = re.match(f"{file_prefix}(\\d+).mp4", file)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1

def save_video(frames, output_dir, frame_rate, file_index):
    if not frames:
        print("No frames to save")
        return

    # Use 'X264' as fourcc for H.264, and change file extension to .mp4
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    output_path = os.path.join(output_dir, f'fight_record_{file_index}.mp4')
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, size)

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")


def put_label_on_frame(frame, label, location=(450, 50), font_scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, label, location, font, font_scale, color, thickness, cv2.LINE_AA)

# Load SSD MobileNet V2 model
ssd_model_dir = 'ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'
ssd_model = tf.saved_model.load(ssd_model_dir)
category_index = label_map_util.create_category_index_from_labelmap('models/research/object_detection/data/mscoco_label_map.pbtxt', use_display_name=True)

#(mm:ss) time format
def format_duration(duration):
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    return f"{minutes:02d}:{seconds:02d}"

def detect_person_ssd(frame, ssd_model, category_index, person_tracker):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor([frame_rgb], dtype=tf.uint8)
    detections = ssd_model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    person_detections_mask = detections['detection_classes'] == 1
    detections['detection_boxes'] = detections['detection_boxes'][person_detections_mask]
    detections['detection_classes'] = detections['detection_classes'][person_detections_mask]
    detections['detection_scores'] = detections['detection_scores'][person_detections_mask]

    frame_with_detections = frame.copy()
    current_time = time.time()
    for i, box in enumerate(detections['detection_boxes']):
        if detections['detection_scores'][i] > 0.6:
            person_id = i
            if person_id not in person_tracker:
                person_tracker[person_id] = {'first_seen': current_time, 'last_seen': current_time, 'duration': 0}
            else:
                person_tracker[person_id]['last_seen'] = current_time
                person_tracker[person_id]['duration'] = current_time - person_tracker[person_id]['first_seen']

            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
            cv2.rectangle(frame_with_detections, start_point, end_point, (0, 255, 0), 2)

            # Displaying person ID at the top-left corner
            cv2.putText(frame_with_detections, f"ID: {person_id}", (start_point[0], start_point[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # Displaying duration at the bottom-left corner
            duration_text = f"Duration: {format_duration(person_tracker[person_id]['duration'])}"
            cv2.putText(frame_with_detections, duration_text, (start_point[0], end_point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame_with_detections

# Load the pre-trained violence detection model
violence_model = load_model('Model_Used/Violence_Model/Violence_model.h5')

        
def sharpen_frame(frame):
    # Define a sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                 [-1, 5,-1],
                                 [0, -1, 0]])
    # Applying the sharpening kernel to the input image
    sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)
    return sharpened_frame


def process_video_stream():
    video_dir = "test_video"
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    output_dir = "static/recorded_fights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    threshold = 0.20 # Define a threshold value for deciding between Violence and NonViolence

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        capture = cv2.VideoCapture(video_path)

        fps = capture.get(cv2.CAP_PROP_FPS)
        pre_buffer_size = int(fps * 10)
        post_record_time = 3
        buffer = deque(maxlen=int(pre_buffer_size + fps * post_record_time))
        frames = []
        frame_count = 0
        queue = []
        predicted_category = "Unknown"
        global violence_detected  # Declare violence_detected as global
        violence_detected = False
        global last_sound_time
        last_sound_time = None
        is_recording = False
        record_countdown = 0
        person_tracker = {}


        file_index = get_next_file_index(output_dir)

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            # Apply sharpening to the frame
            frame = sharpen_frame(frame)

            frame_count += 1

            if frame_count % 2 != 0:  # Frame Skipping Rate
                continue

            frame = cv2.resize(frame, (640, 480))
            frame_with_detections = detect_person_ssd(frame, ssd_model, category_index, person_tracker)

            resized_frame = cv2.resize(frame_with_detections, (90, 90))
            frames.append(resized_frame)
            buffer.append(frame_with_detections)

            if len(frames) >= 40:  # Number of frames for violence detection
                sequence = np.array(frames)
                prediction = violence_model.predict(np.expand_dims(sequence, axis=0))
                queue.append(prediction)
                frames = []
                frame_count = 0

                if queue:
                    results = np.array(queue).mean(axis=0)

                    # Ensure that results array is not empty
                    if results.size == 0:
                        print("No predictions available.")
                        continue

                    if results.ndim == 1:
                        confidence = results[0]  # Confidence score for the only class
                    else:
                        confidence = results[0, 1]  # Confidence score for violence class

                    # Adjust sensitivity based on confidence level
                    if confidence >= threshold:
                        print(f'Predicted: Violence (Confidence: {confidence:.2f})')
                        predicted_category = 'Violence'
                    else:
                        print(f'Predicted: NonViolence (Confidence: {1 - confidence:.2f})')
                        predicted_category = 'NonViolence'

                    queue = []

                    if predicted_category == "Violence" and not is_recording:
                        is_recording = True
                        record_countdown = fps * post_record_time
                        file_index += 1
                        violence_detected = True  # Set violence detection flag to True
                        last_sound_time = time.time()  # Update last_sound_time

            color = (0, 255, 0) if predicted_category == "NonViolence" else (0, 0, 255)
            put_label_on_frame(frame_with_detections, predicted_category, location=(450, 50), font_scale=1, color=color)

            if is_recording:
                record_countdown -= 1
                if record_countdown <= 0:
                    is_recording = False
                    save_video(list(buffer), output_dir, fps, file_index)
                    buffer.clear()
                    file_index = get_next_file_index(output_dir)

            # Convert frame to JPEG format for streaming
            ret, jpeg_buffer = cv2.imencode('.jpg', frame_with_detections)
            frame_bytes = jpeg_buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # Send violence detection flag in the response
            yield f"violence_detected:{violence_detected}\n".encode()

        capture.release()
