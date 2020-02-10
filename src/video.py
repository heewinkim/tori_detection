
import cv2
from .tori_detection import ToriDetection

cap = cv2.VideoCapture(0)
tori = ToriDetection()

while True:

    _,img = cap.read()

    confidence = tori._predict(img)

    cv2.putText(img,'confidence : {}%'.format(round(confidence*100)),(15,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255))

    if confidence >= 0.75:
        cv2.putText(img,'tori appear!',(img.shape[1]//2-30,img.shape[0]//2-15),cv2.FONT_HERSHEY_COMPLEX,2.0,(0,0,255),2)

    img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    cv2.imshow('frame',img)

    if cv2.waitKey(1)==ord('q'):
        break




