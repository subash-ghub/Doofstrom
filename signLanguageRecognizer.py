from landmarkDetection import *
from decimal import Decimal
# from win32com.client import Dispatch
from time import sleep
import pyttsx3
engine = pyttsx3.init()
    

def signLanguageRecognizer():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.load_weights('G:/signbot/MODEL/action1.h5')
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.2
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            image, results = mediapipeDetection(frame, holistic)

            keypoints = extractKeypoints(results)

            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = model.predict(tf.expand_dims(sequence, axis=0))[0]
                ind = np.argpartition(res, -3)[-3:]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if (max(res) >= threshold):
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                        cv2.putText(image,
                                        actions[np.argmax(res)] + ' : ' + str((max(res) * 100).astype(float)) + ' %',
                                        (15, 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        print(actions[ind[0]] + ':' + str((res[ind[0]] * 100).astype(float)) + ' %' + '      '
                                  + actions[ind[1]] + ':' + str((res[ind[1]] * 100).astype(float)) + ' %' + '      '
                                  + actions[ind[2]] + ':' + str((res[ind[2]] * 100).astype(float)) + ' %')

                        # cv2.putText(image, actions[np.argmax(res)] + ' : ' + str((max(res) * 100).astype(float)) + ' %',
                        #             (15, 25),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                        # speak = Dispatch("SAPI.SpVoice")
                        engine.say(actions[np.argmax(res)])
                        engine.runAndWait()
                        # speak.Speak(actions[np.argmax(res)])
                        # continue

            cv2.imshow("window", image)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

        cap.release()
        cv2.destroyAllWindows()
