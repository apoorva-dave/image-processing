from skimage import io as skio
url = 'http://i.stack.imgur.com/SYxmp.jpg'
img = skio.imread(url)

print("shape of image: {}".format(img.shape))
print("dtype of image: {}".format(img.dtype))

from skimage import filters
sobel = filters.sobel(img)


import matplotlib.pyplot as plt

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200

plt.imshow(sobel)
plt.show()

blurred = filters.gaussian(sobel, sigma=2.0)
plt.imshow(blurred)
plt.show()

import numpy as np
light_spots = np.array((img > 245).nonzero()).T

print(light_spots.shape)

plt.plot(light_spots[:, 1], light_spots[:, 0], 'o')
plt.imshow(img)
plt.title('light spots in image')
plt.show()

dark_spots = np.array((img < 3).nonzero()).T
print(dark_spots.shape)

plt.plot(dark_spots[:, 1], dark_spots[:, 0], 'o')
plt.imshow(img)
plt.title('dark spots in image')
plt.show()


from scipy import ndimage as ndi
bool_mask = np.zeros(img.shape, dtype=np.bool)
bool_mask[tuple(light_spots.T)] = True
bool_mask[tuple(dark_spots.T)] = True
seed_mask, num_seeds = ndi.label(bool_mask)
print(num_seeds)

from skimage import morphology
ws = morphology.watershed(blurred, seed_mask)
plt.imshow(ws)
plt.show()

background = max(set(ws.ravel()), key=lambda g: np.sum(ws == g))
print(background)

background_mask = (ws == background)
print(background_mask)

plt.imshow(~background_mask)
plt.show()

cleaned = img * ~background_mask
plt.imshow(cleaned)
plt.show()
