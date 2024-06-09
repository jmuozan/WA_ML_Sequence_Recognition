import os
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
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

# Global variables to store the selected model and data
model = None
data = None
actions = []
label_map = {}
reverse_label_map = {}
stop_event = threading.Event()

def get_model_options():
    # Scan the models directory for available subfolders
    subfolders = [f.name for f in os.scandir(MODELS_DIR) if f.is_dir()]
    return subfolders

def initialize_model_and_data(selected_model):
    global model, data, actions, label_map, reverse_label_map
    model_path = os.path.join(MODELS_DIR, selected_model, f'{selected_model}.h5')
    csv_path = os.path.join(MODELS_DIR, selected_model, f'{selected_model}.csv')
    
    data = pd.read_csv(csv_path)
    labels = data.iloc[:, 0]
    actions = list(set(labels))
    label_map = {label: num for num, label in enumerate(actions)}
    model = load_action_model(model_path)
    reverse_label_map = {value: key for key, value in label_map.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn', methods=['GET', 'POST'])
def learn():
    model_options = get_model_options()
    if request.method == 'POST':
        selected_model = request.form['model']
        initialize_model_and_data(selected_model)
        return redirect(url_for('video_feed'))
    return render_template('learn.html', model_options=model_options)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/create')
def create():
    return render_template('create.html')

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

            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = predict_action(model, sequence)
                    action = actions[np.argmax(res)]
                    incorrect_movement = action.endswith("_W")
                    draw_styled_landmarks(image, results, incorrect_movement)
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

@app.route('/video_feed')
def video_feed():
    if model is None or data is None:
        return redirect(url_for('learn'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_prediction', methods=['POST'])
def stop_prediction():
    global stop_event
    stop_event.set()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
