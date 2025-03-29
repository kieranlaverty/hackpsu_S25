import cv2
import torch
import numpy as np 
from facenet_pytorch import MTCNN

class FaceDetector(object):

    def __init__(self ,mtcnn):
        self.mtcnn = mtcnn
    
    def _draw(self, frame, boxes, probs, landmarks):
        try:
            for box, prob, ld in zip(boxes, probls, landmarks):
                cv2.rectangle(frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (0, 0, 255), 
                            thickeness = 2)

                cv2.putText(frame, str(prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

def run(self):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        try:
            box, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

            self._draw(frame, boxes, probs, landmarks)

        except:
            pass

        
        cv2.imshow('Face Detection', frame)

        if vc2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run()
