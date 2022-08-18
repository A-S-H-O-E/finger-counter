import cv2
import mediapipe as mp
cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)
tipid = [4, 8, 12, 16, 20]


def drawhand(img, handpoints):
    if(handpoints):
        for h in handpoints:
            mp_drawing.draw_landmarks(img, h, mp_hands.HAND_CONNECTIONS)

def cf(img,handpoints,handnumber = 0):
    if(handpoints):
        allpoints = handpoints[handnumber].landmark
        finger = []
        for t in tipid:
            fingerty = allpoints[t].y
            fingerbtmy = allpoints[t-2].y
            if(fingerbtmy > fingerty):
                finger.append(1)
            if(fingerbtmy < fingerty):
                finger.append(0)
        totalfinger = finger.count(1)
        text = f'Fingers:{totalfinger}'   
        cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(170,45,85),2)


while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    results = hands.process(img)
    handpoints = results.multi_hand_landmarks
    drawhand(img, handpoints)
    cf(img,handpoints)
    cv2.imshow('dp',img)
    if(cv2.waitKey(1) == 32):
        break
cv2.destroyAllWindows()