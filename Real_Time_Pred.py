import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pandas as pd

data = pd.read_csv('modified_file.csv')
labels = data.iloc[:, 0]
actions = list(set(labels))
print(actions)
label_map = {label:num for num, label in enumerate(actions)}
label_map

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model_path = 'actions.h5'
model = load_model(model_path)

reverse_label_map = {value: key for key, value in label_map.items()}

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results, incorrect_movement):
    if incorrect_movement:
        color = (0, 0, 255) 
    else:
        color = (0, 255, 0)  

    pose_spec = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4)
    hand_spec = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            pose_spec, pose_spec
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            hand_spec, hand_spec
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            hand_spec, hand_spec
        )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

sequence = []
sentence = []
threshold = 0.5

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
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