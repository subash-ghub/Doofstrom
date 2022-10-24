import cv2
import numpy as np
import mediapipe as mp
import os
import time
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_PATH = os.path.join('G:/signbot/DATA/MP_DATA')
DATA_PATH_VIDEO = os.path.join('G:/signbot/DATA/MP_VIDEOS')
actions = np.array(open('G:/signbot/DATA/classes.txt', 'r').read().split('\n'))
number_of_sequences = 100
every_sequence_length = 30

mpHolistic = mp.solutions.holistic
mpDrawing = mp.solutions.drawing_utils

def mediapipeDetection(image_param, model):
    image = cv2.cvtColor(image_param, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def drawLandmarks(image, results):
    mpDrawing.draw_landmarks(image, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
                             mpDrawing.DrawingSpec(color=(81, 23, 11), thickness=2, circle_radius=4),
                             mpDrawing.DrawingSpec(color=(81, 45, 122), thickness=2, circle_radius=2))
    mpDrawing.draw_landmarks(image, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
                             mpDrawing.DrawingSpec(color=(122, 23, 77), thickness=2, circle_radius=4),
                             mpDrawing.DrawingSpec(color=(122, 45, 249), thickness=2, circle_radius=2))

    mpDrawing.draw_landmarks(image,results.face_landmarks, mpHolistic.FACEMESH_TESSELATION,
                             mpDrawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mpDrawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

def extractKeypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    return np.concatenate([lh, rh])
