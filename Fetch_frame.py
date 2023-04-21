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
    bf = cv2.BFMatcher()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Find the top 20 corners using the cv2.goodFeaturesToTrack()
    feats = cv2.goodFeaturesToTrack(gray, 1000, qualityLevel = 0.01, minDistance = 10)

    #np.mean(img, axis=2).astype(np.uint8)
    return feats
def process_frame(img, img_old):
    img = cv2.resize(img, (W, H))
    img_old = cv2.resize(img, (W, H))
    kp = extractor(img)
    # Iterate over the corners and draw a circle at that location
    for p in kp:
        u, v = map(lambda x: int(round(x)), p[0])
        cv2.circle(img, (u, v), color= (0, 255, 0), radius = 3)
    kp1, des1 = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(img_old, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img, kp1, img_old, kp2, matches[:3000], img, flags=2)
    cv2.imshow('Data association', img3)
    cv2.waitKey()
    #disp.paint(img3)

if __name__ == "__main__":
    cap = cv2.VideoCapture("pexels_2.mp4")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True and count is not 0:
            process_frame(frame, frame_old)
        elif ret == True and count is 0:
            frame_old = frame
            count = count + 1
        else:
            break
        
        