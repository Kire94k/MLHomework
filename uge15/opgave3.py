import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("uge15/beach.jpg")
plt.figure(figsize=(10,10))
plt.imshow(original_image)

image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
pixels_values = image.reshape((-1, 3))
pixels_values = np.float32(pixels_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

K = 5

_, labels, (centers) = cv2.kmeans(pixels_values, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

plt.figure(figsize=(10,10))
plt.imshow(segmented_image)
plt.show()