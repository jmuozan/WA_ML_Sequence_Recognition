import os
import time
import csv
from flask import Flask, render_template, request, Response, jsonify
import threading
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from modules.detection import mediapipe_detection, draw_styled_landmarks
from modules.keypoints import extract_keypoints
from modules.model_handler import load_action_model, predict_action

app = Flask(__name__)

# Path to the models directory
MODELS_DIR = 'models'
VID_DIR = 'created'

# Global variables to store the selected model and data
model = None
data = None
actions = []
label_map = {}
reverse_label_map = {}
stop_event = threading.Event()
recording_event = threading.Event()
all_movements_done_event = threading.Event()
current_movement = 0
current_repetition = 0
current_status = ""

def get_model_options():
    subfolders = [f.name for f in os.scandir(MODELS_DIR) if f.is_dir()]
    return subfolders

def initialize_model_and_data(selected_model):
    global model, data, actions, label_map, reverse_label_map
    model_path = os.path.join(MODELS_DIR, selected_model, f'{selected_model}.h5')
    csv_path = os.path.join(MODELS_DIR, selected_model, f'{selected_model}.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        model = None
        data = None
        actions = []
        label_map = {}
        reverse_label_map = {}
        print(f"Error: Model or CSV file not found for {selected_model}")
        return False

    data = pd.read_csv(csv_path)
    labels = data.iloc[:, 0]
    actions = list(set(labels))
    label_map = {label: num for num, label in enumerate(actions)}
    model = load_action_model(model_path)
    reverse_label_map = {value: key for key, value in label_map.items()}
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn', methods=['GET', 'POST'])
def learn():
    model_options = get_model_options()
    return render_template('learn.html', model_options=model_options)

@app.route('/select_model', methods=['POST'])
def select_model():
    selected_model = request.json['model']
    if initialize_model_and_data(selected_model):
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Model or CSV file not found'})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/create')
def create():
    return render_template('create.html')

@app.route('/submit_dataset', methods=['POST'])
def submit_dataset():
    craft_name = request.form['craft_name']
    description = request.form['description']
    movements = int(request.form['movements'])
    prize = request.form['prize']
    movement_length = int(request.form['movement_length'])
    repetitions = int(request.form['repetitions'])
    
    print(f'Craft Name: {craft_name}')
    print(f'Description: {description}')
    print(f'Movements: {movements}')
    print(f'Prize: {prize}')
    print(f'Movement Length: {movement_length}')
    print(f'Repetitions: {repetitions}')
    
    return jsonify({'success': True, 'craft_name': craft_name, 'movements': movements, 'movement_length': movement_length, 'repetitions': repetitions})

def generate_frames():
    cap = cv2.VideoCapture(0)
    sequence = []
    threshold = 0.5
    
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame could not be read.")
                break

            frame = cv2.flip(frame, 1)

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results, False)

            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30 and model is not None:
                    res = predict_action(model, sequence)
                    if res is not None:
                        action = actions[np.argmax(res)]
                        incorrect_movement = action.endswith("_W")
                        color = (0, 0, 255) if incorrect_movement else (0, 255, 0)
                        cv2.putText(image, f'Action: {action}', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except GeneratorExit:
                cap.release()
                return

    cap.release()

def record_video(length, craft_name, repetitions, movements):
    global current_movement, current_repetition, current_status
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open video capture")
        return
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Video Resolution: {width}x{height} at {fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_dir = os.path.join(VID_DIR, craft_name)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    for movement in range(movements):
        movement_dir = os.path.join(video_dir, f'Movement_{movement + 1}')
        if not os.path.exists(movement_dir):
            os.makedirs(movement_dir)
        current_movement = movement + 1

        for i in range(repetitions):
            current_repetition = i + 1
            video_path = os.path.join(movement_dir, f'{craft_name}_movement_{movement + 1}_rep_{i + 1}.avi')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print("Error: Could not open video writer")
                return

            start_time = time.time()
            current_status = f'Recording Movement {current_movement}, Repetition {current_repetition}'
            while time.time() - start_time < length:
                success, frame = cap.read()
                if not success:
                    print("Error: Unable to read frame from video capture")
                    break
                frame = cv2.flip(frame, 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)
                frame_rgb.flags.writeable = True
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                draw_styled_landmarks(frame, results, False)

                # Write every frame
                out.write(frame)

            out.release()
            print(f"Video saved: {video_path}")

        if movement < movements - 1:
            current_status = f'Movement {movement + 1} completed. Preparing for movement {movement + 2}...'
            print(current_status)
            time.sleep(5)

    cap.release()
    cv2.destroyAllWindows()
    all_movements_done_event.set()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data = request.get_json()
    length = int(data['length'])
    craft_name = data['craft_name']
    repetitions = int(data['repetitions'])
    movements = int(data['movements'])
    all_movements_done_event.clear()
    recording_thread = threading.Thread(target=record_video, args=(length, craft_name, repetitions, movements))
    recording_thread.start()
    return jsonify({'success': True})

@app.route('/stop_prediction', methods=['POST'])
def stop_prediction():
    global stop_event
    stop_event.set()
    return jsonify({'success': True})

@app.route('/check_status', methods=['POST'])
def check_status():
    global current_movement, current_repetition, current_status
    if all_movements_done_event.is_set():
        return jsonify({'all_movements_done': True})
    return jsonify({'all_movements_done': False, 'current_movement': current_movement, 'current_repetition': current_repetition, 'current_status': current_status})

def process_videos_and_create_csv():
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    header = ['label', 'sequence']
    for i in range(33):  # pose landmarks
        header += [f'pose_{i}_x', f'pose_{i}_y', f'pose_{i}_z']
    for i in range(21):  # left hand landmarks
        header += [f'left_hand_{i}_x', f'left_hand_{i}_y', f'left_hand_{i}_z']
    for i in range(21):  # right hand landmarks
        header += [f'right_hand_{i}_x', f'right_hand_{i}_y', f'right_hand_{i}_z']

    craft_name = ''
    video_dir = ''
    
    for root, dirs, files in os.walk(VID_DIR):
        for dir in dirs:
            craft_name = dir
            video_dir = os.path.join(VID_DIR, craft_name)
            break
        break
    
    if not craft_name:
        return
    
    craft_model_dir = os.path.join(MODELS_DIR, craft_name)
    if not os.path.exists(craft_model_dir):
        os.makedirs(craft_model_dir)

    with open(os.path.join(craft_model_dir, 'dataset.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for root, dirs, files in os.walk(video_dir):
            for dir in dirs:
                label = dir
                dir_path = os.path.join(root, dir)
                sequence = 0
                for video_file in os.listdir(dir_path):
                    if video_file.endswith('.avi'):
                        video_path = os.path.join(dir_path, video_file)
                        cap = cv2.VideoCapture(video_path)
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = holistic.process(frame_rgb)
                            keypoints = extract_keypoints(results)
                            keypoints = [np.float32(0) if np.isnan(x) else np.float32(x) for x in keypoints]
                            row = [label, sequence] + keypoints
                            writer.writerow(row)
                        cap.release()
                        sequence += 1

@app.route('/start_training', methods=['POST'])
def start_training():
    process_videos_and_create_csv()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
