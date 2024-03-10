import os

import cv2
import numpy as np

from utils import get_face_landmarks


data_dir = "emotion-recognition-python-scikit-learn-mediapipe-main/data/train"
# data_dir = "Emotion_detection_with_CNN-main/data/train"

output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        image = cv2.imread(image_path)

        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))  # add labels
            output.append(face_landmarks)
    print(f"Finish {emotion}")

np.savetxt(
    "emotion-recognition-python-scikit-learn-mediapipe-main/data.txt",
    np.asarray(output),
)
