import cv2
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as tfk

from pygame import mixer


def get_frame(capture: cv2.VideoCapture):
    """Gets frame from firt webcam.

    Args:
        capture (cv2.VideoCapture): video capture from first webcam

    Returns:
        _type_: one frame
    """
    _, frame = capture.read()
    return frame


def detect_face(frame, xml_file: str):
    """Detects face.

    Args:
        frame (_type_): one frame
        xml_file (str): xml files for haar cascade

    Returns:
        _type_: frame with face with bounding box
        _type_: gray scale frame
    """
    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(xml_file)
    face = face_detector.detectMultiScale(gray_scale_frame, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return frame, gray_scale_frame


def detect_eye(frame, gray_scale_frame, model: tfk.Model, xml_file: str) -> str:
    """Detects eye.

    Args:
        frame (_type_): one frame
        gray_scale_frame (_type_): gray scale frame
        model (tfk.Model): trained and saved model
        xml_file (str): xml haar cascade file to detect eye

    Returns:
        str: label for one eye
    """
    eye_detector = cv2.CascadeClassifier(xml_file)
    eye = eye_detector.detectMultiScale(gray_scale_frame)
    for x, y, w, h in eye:
        extracted_eye = frame[y: y + h, x: x + w]
        gray_eye = cv2.cvtColor(extracted_eye, cv2.COLOR_BGR2GRAY)
        reszied_eye = cv2.resize(gray_eye, (24, 24))
        rescaled = (1 / 255) * reszied_eye
        reshaped = rescaled.reshape(24, 24, -1)
        expanded = np.expand_dims(reshaped, axis=0)
        pred = model.predict(expanded)
        if np.argmax(pred) == 1:
            label = 'open'
        elif np.argmax(pred) == 0:
            label = 'closed'
    return label


def get_drawsiness_score(right_eye_label: str, left_eye_label: str, frame,
                         height: int, font) -> int:
    """Gets drawsiness score.

    Args:
        right_eye_label (str): right eye label
        left_eye_label (str): left eye label
        frame (_type_): one frame
        height (int): height of the frame
        font (_type_): font

    Returns:
        int: score
    """
    score = 0
    if right_eye_label == 'closed' and left_eye_label == 'closed':
        score += 1
        cv2.putText(frame, 'Closed', (10, height - 20), font, 1, (255, 255, 255),
                    1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, 'Open', (10, height - 20), font, 1, (255, 255, 255),
                    1, cv2.LINE_AA)
    if score < 0:
        score = 0
    return score


def main():
    capture = cv2.VideoCapture(0)
    model = tfk.models.load_model('model_path')
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    thickness = 2
    mixer.init()
    sound = mixer.Sound('/kaggle/input/warning/alarm.wav')
    while True:
        frame = get_frame(capture)
        frame, gray_scale_frame = detect_face(frame, 'haarcascade_frontalface_alt.xml')
        left_label = detect_eye(frame, gray_scale_frame, model, 'haarcascade_lefteye_2splits.xml')
        right_label = detect_eye(frame, gray_scale_frame, model, 'haarcascade_righteye_2splits.xml')
        score = get_drawsiness_score(right_label, left_label, frame, height, width, font)
        if score > 7:
            cv2.putText(frame, 'Score' + str(score), (100, height - 20), font, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)
            fig = plt.figure()
            plt.imshow(frame)
            plt.axis('off')
            plt.savefig('closed_eye_frame.png')
            if thickness > 16:
                thickness -= 2
            else:
                thickness += 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), thickness)
            sound.play()
            time.sleep(1)


if __name__ == '__main__':
    main()