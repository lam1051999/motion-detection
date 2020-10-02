import numpy as np
import cv2 as cv
import imutils

im = cv.imread('male3.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

print("the number of contours: ", len(contours))
cv.drawContours(im, contours, -1, (0, 255, 0), 3)
cv.imshow("imgae: ", im)
cv.imshow("gray: ", imgray)
cv.waitKey(0)
cv.destroyAllWindows()
