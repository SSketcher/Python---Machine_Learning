from cv2 import cv2
import numpy as np
import dlib
import data_processing as dp
import yawning_classifier as yc


cap = cv2.VideoCapture('resources\YawningPeople480p.avi')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models\shape_predictor_68_face_landmarks.dat")

data = dp.Data()

frames_counter = 0

if cap.isOpened():
    width  = round(cap.get(3))
    height = round(cap.get(4))

while True:
    frames_counter += 1
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, str(frames_counter), (width-50, height-30), font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    for face in faces:
        landmarks = predictor(gray, face)
        data.features_data(landmarks)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0),  1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(3000)
    data.set_output(key)

    if key == 27:
        break

data.makeCSV()
cap.release()
cv2.destroyAllWindows()

#yc.svm_learn(np.array(data.data))