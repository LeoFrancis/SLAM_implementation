import cv2
from display import Display
import numpy as np
W = 1920//2
H = 1080//2
GX = 16
GY = 16

disp = Display(W, H)
orb = cv2.ORB_create()


def extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Find the top 20 corners using the cv2.goodFeaturesToTrack()
    feats = cv2.goodFeaturesToTrack(gray, 1000, qualityLevel = 0.01, minDistance = 10)
    #np.mean(img, axis=2).astype(np.uint8)
    return feats
def process_frame(img):
    img = cv2.resize(img, (W, H))
    kp = extractor(img)
    # Iterate over the corners and draw a circle at that location
    for p in kp:
        u, v = map(lambda x: int(round(x)), p[0])
        cv2.circle(img, (u, v), color= (0, 255, 0), radius = 3)
    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("pexels_1.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
        
        