import cv2
import numpy as np 

def detect_color(img):
    #convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #define the range of blue color in HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    #Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #bitwise-AND mask and the original image
    res = cv2.bitwise_and(img, img, mask = mask)


    return res, mask

