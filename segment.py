import numpy as np
import cv2


img = cv2.imread('baby.jpg')
fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(img)


cv2.imshow('image', fgmask)
cv2.waitKey(0)
cv2.destroyAllWindows()
