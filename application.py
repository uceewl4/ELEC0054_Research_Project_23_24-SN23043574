# here put the import lib
import os
import cv2
import smtplib
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
from email.header import Header
from email.mime.text import MIMEText
from streamlit_option_menu import option_menu
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import Model
import cv2
import numpy as np
from keras.models import model_from_json

from utils import get_features, load_data, load_model
import cv2
import mediapipe as mp
import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import argparse
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
# parser = argparse.ArgumentParser(description="Argparse")
# parser.add_argument("--app", type=str, default="speech", help="speech/video/realtime")
# args = parser.parse_args()
app = input("Please select your choice for emotion detection: [1] Video, [2] Real-time")


def get_face_landmarks(image, draw=False, static_image_mode=True):

    # Read the input image
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )
    image_rows, image_cols, _ = image.shape
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:

        if draw:

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = []
        ys_ = []
        zs_ = []
        for idx in ls_single_face:  # every single landmark get three coordinates xyz
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)
        for j in range(len(xs_)):  # get landmarks of the face
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))
    face_mesh.close()

    return image_landmarks


# if app == "0":
#     # Record audio for 5 seconds (adjust duration as needed)
#     duration = 5  # seconds
#     fs = 24414  # Sample rate, as TESS
#     print("Start recording...")
#     recording = sd.rec(int(5 * fs), samplerate=fs, channels=1)
#     sd.wait()  # Wait until recording is finished
#     print("Finish recording.")
#     if not os.path.exists("outputs/application/"):
#         os.makedirs("outputs/application/")
#     wav.write("outputs/application/audio.wav", fs, recording)

#     emotion_dict = {
#         0: "angry",
#         1: "disgust",
#         2: "fear",
#         3: "happy",
#         4: "neutral",
#         5: "ps",
#         6: "sad",
#     }

#     feature, X = get_features(
#         "TESS",
#         "AlexNet",
#         "outputs/application/audio.wav",
#         "mfcc",
#         n_mfcc=40,
#         n_mels=128,
#         max_length=150,
#         sr=fs,
#     )
#     feature = np.array(feature)
#     print(feature.shape)

#     emotion_model = tf.keras.models.load_model("outputs/speech/models/AlexNet.h5")
#     print("Loaded model from disk")

#     emotion_prediction = emotion_model.predict(feature.reshape(40, 150))
#     maxindex = int(np.argmax(emotion_prediction))


emotion_dict = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Sadness",
    6: "Surprise",
}

emotion_model = tf.keras.models.load_model("outputs/image/models/CNN.h5")
print("Loaded model from disk")

if app == "1":
    path = input("Please select your video path...")
    cap = cv2.VideoCapture(path)

elif app == "2":
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    if not ret:
        break
    face_detector = cv2.CascadeClassifier(
        "/Users/anlly/Desktop/ucl/Final_Project/ELEC0054_Research_Project_23_24-SN23043574/outputs/image/models/haarcascades/haarcascade_frontalface_default.xml",
    )  # crop the face
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5
    )

    # print(num_faces)
    # take each face available on the camera and Preprocess it
    for x, y, w, h in num_faces:  # face bounding boxes, rectangle
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[
            y : y + h, x : x + w
        ]  # crop each face into one gray frame
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0
        )  # resize into the images

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        print(maxindex)
        cv2.putText(
            frame,
            emotion_dict[maxindex],
            (x + 5, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
