import numpy as np
import pandas as pd
import dlib

class Data():
    def __init__(self):
        self.data = []
        self.labels = []


    def features_data(self, landmarks):
        if len(landmarks.parts()) < 68:
            return "error"
        frame = []
        #mouth features
        top_lip = self._get_lanmark_points(landmarks, 61, 64)
        botom_lip = self._get_lanmark_points(landmarks, 65, 68)
        botom_lip.reverse()
        mouth_opening = []
        for i in range(len(top_lip)):
            mouth_opening.append(self._calc_distance(top_lip[i], botom_lip[i]))
        mouth_with = self._calc_distance(self._point2tuple(landmarks.part(61)), self._point2tuple(landmarks.part(65)))
        frame.append(np.average(mouth_opening)/mouth_with)
        #left eye features
        top_left_eyelid = self._get_lanmark_points(landmarks, 37, 39)
        botom_left_eyelid = self._get_lanmark_points(landmarks, 40, 42)
        botom_left_eyelid.reverse()
        left_eye_opening = []
        for j in range(len(top_left_eyelid)):
            left_eye_opening.append(self._calc_distance(top_left_eyelid[j], botom_left_eyelid[j]))
        left_eye_with = self._calc_distance(self._point2tuple(landmarks.part(37)), self._point2tuple(landmarks.part(40)))
        frame.append(np.average(left_eye_opening)/left_eye_with)
        #right eye features
        top_right_eyelid = self._get_lanmark_points(landmarks, 43, 45)
        botom_right_eyelid = self._get_lanmark_points(landmarks, 46, 48)
        botom_right_eyelid.reverse()
        right_eye_opening = []
        for k in range(len(top_right_eyelid)):
            right_eye_opening.append(self._calc_distance(top_right_eyelid[k], botom_right_eyelid[k]))
        right_eye_with = self._calc_distance(self._point2tuple(landmarks.part(43)), self._point2tuple(landmarks.part(46)))
        frame.append(np.average(right_eye_opening)/right_eye_with)
        self.data.append(np.array(frame))
        return(np.array(frame))

    
    def set_output(self, valeu):
        if valeu == 48:
            self.labels.append(0)
        elif valeu == 49:
            self.labels.append(1)
        elif valeu == 27:
            self.data.pop()
        else:
            self.data.pop()
        


    def makeCSV(self, path = 'yawning_data.csv'):
        try:
            head = ['mouth', 'left_eye', 'right_eye']
            df = pd.DataFrame(np.array(self.data), columns = head)
            df['Labels'] = self.labels
            df.to_csv(path, index = False)
        except:
            return 'error'


    def openCSV(self, path = 'yawning_data.csv'):
        try:
            df = pd.read_csv(path, header = 0)
            data = df.to_numpy()
            return(data)
        except:
            return 'error'


    def _get_lanmark_points(self, landmarks, down_limit, up_limit):
        temp = []
        for i in range(down_limit, up_limit):
            temp.append((landmarks.part(i).x, landmarks.part(i).y))
        return temp


    def _calc_distance(self, point1, point2):
        return np.sqrt(np.power(point2[0]-point1[0], 2) + np.power(point2[-1]-point1[-1], 2))


    def _point2tuple(self, point):
        return (point.x, point.y)