import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("C:\\Users\\khjen\\OneDrive\\ML-homework\\MLHomework\\uge15\\Hundehvalpe2_dk.jpg")
plt.figure(figsize=(10,10))
plt.imshow(original_image)
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)