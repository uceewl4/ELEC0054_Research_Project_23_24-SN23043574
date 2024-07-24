# -*- encoding: utf-8 -*-
"""
@File    :   application.py
@Time    :   2024/07/24 06:18:12
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes implementation process of application for video capture and real-time camera.
            Notice that this file is separate from the main.py file and has to be run individually.
            The code refers to https://github.com/oarriaga/face_classification/tree/master, https://github.com/datamagic2020/Emotion_detection_with_CNN.
"""


# here put the import lib
import cv2
import warnings
import numpy as np
import mediapipe as mp
import tensorflow as tf


warnings.filterwarnings("ignore")


def get_face_landmarks(image, draw=False, static_image_mode=True):
    """
    description: This function is used for extracting face landmarks from faces detected.
    param {*} image: input frame
    return {*}: image landmarks
    """

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
        for (
            idx
        ) in (
            ls_single_face
        ):  # every single landmark get three coordinates xyz with 468 landmarks in total
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)
        for j in range(
            len(xs_)
        ):  # get landmark as relative distance with smallest localization
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))
    face_mesh.close()

    return image_landmarks


"""
    1 means applying model on selected video and corresponding video path has to be provided.
    2 means real-time capture of facial changing with camera on your PC for emotion detection.
"""
app = input("Please select your choice for emotion detection: [1] Video, [2] Real-time")

# emotion mapping
emotion_dict = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Sadness",
    6: "Surprise",
}

# trained model of CNN for image emotion detection frame by frame
emotion_model = tf.keras.models.load_model("outputs/image/models/CNN.h5")
print("Loaded model from disk")

if app == "1":  # video
    path = input("Please select your video path...")
    cap = cv2.VideoCapture(path)

elif app == "2":  # real-time capture
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

while True:  # detecting continuously
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    # Find haar cascade to draw bounding box around face
    face_detector = cv2.CascadeClassifier(
        "/Users/anlly/Desktop/ucl/Final_Project/ELEC0054_Research_Project_23_24-SN23043574/outputs/image/models/haarcascades/haarcascade_frontalface_default.xml",
    )
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5
    )
    # print(num_faces)

    # process face within the frame
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
        # print(maxindex)

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

        # get face landmarks
        face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    # show frames
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
