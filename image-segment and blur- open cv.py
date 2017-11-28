import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('baby.jpg')
img_copy = img.copy()
b, g, r = cv2.split(img_copy)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(gray.shape)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('image',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)


cv2.imshow('image',sure_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)


cv2.imshow('image',sure_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers += 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

img[markers == -1] = [250,0,0]
#
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

background = max(set(markers.ravel()), key=lambda g: np.sum(markers == g))
print(background)
print(background.shape)

background_mask = (markers == background)
plt.imshow(~background_mask)
plt.gray()
plt.show()

blur = cv2.GaussianBlur(img, (19, 19), 0)
blur_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
print(blur_gray.shape)
blur_rgb = cv2.cvtColor(blur_gray, cv2.COLOR_GRAY2RGB)
print(blur_rgb.shape)


cleaned = blur_gray * background_mask
cleaned2 = gray * ~background_mask

cleaned3 = cleaned + cleaned2
plt.imshow(cleaned3)
plt.show()
cleaned3 = cv2.merge((b,g,r))
plt.imshow(cleaned3)
plt.show()










