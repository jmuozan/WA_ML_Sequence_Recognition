from tensorflow.keras.models import load_model
import numpy as np

def load_action_model(model_path):
    return load_model(model_path)

def predict_action(model, sequence):
    return model.predict(np.expand_dims(sequence, axis=0))[0]