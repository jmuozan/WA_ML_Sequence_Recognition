from flask import Flask, render_template, Response
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from modules.detection import mediapipe_detection, draw_styled_landmarks
from modules.keypoints import extract_keypoints
from modules.model_handler import load_action_model, predict_action

app = Flask(__name__)

# Load model and other required data
data = pd.read_csv('modified_file.csv')
labels = data.iloc[:, 0]
actions = list(set(labels))
label_map = {label: num for num, label in enumerate(actions)}

model_path = 'actions.h5'
model = load_action_model(model_path)

reverse_label_map = {value: key for key, value in label_map.items()}

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    sequence = []
    threshold = 0.5
    
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
