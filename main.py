import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from modules.detection import mediapipe_detection, draw_styled_landmarks
from modules.keypoints import extract_keypoints
from modules.model_handler import load_action_model, predict_action

# Load data and model
data = pd.read_csv('modified_file.csv')
labels = data.iloc[:, 0]
actions = list(set(labels))
label_map = {label:num for num, label in enumerate(actions)}

model_path = 'actions.h5'
model = load_action_model(model_path)

reverse_label_map = {value: key for key, value in label_map.items()}

sequence = []
sentence = []
threshold = 0.5

cap = cv2.VideoCapture(0)
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
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

        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()