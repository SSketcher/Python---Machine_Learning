from cv2 import cv2
from sklearn import svm
from joblib import dump, load
import numpy as np
import dlib
import data_processing as dp


cap = cv2.VideoCapture('resources\YawningPeople480p.avi')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models\shape_predictor_68_face_landmarks.dat")

clf = load('models\yawing_classifier.joblib')

data = dp.Data()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        pred = 0
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks = predictor(gray, face)
        features = data.features_data(landmarks)
        pred = clf.predict([features])
        print(features)
        print(pred)
        if pred == 1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Ziewa', (round(x1+(x2-x1)/2), y1-2), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break


