import cv2
from display import Display
from extractor import Extractor
import numpy as np

#Constants
W = 1920//2
H = 1080//2

#Init/Constructors
disp = Display(W, H)
fe = Extractor()
def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches = fe.extractor(img)
    if matches is None:
        return

    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), color = (0,255,0), radius = 3)
        cv2.line(img, (u1, v1), (u2, v2), color = (255,0,0))

    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("pexels_2.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
        
        