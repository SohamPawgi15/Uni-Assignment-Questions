import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


# Q.1) a)

image_path = "images/elvis.bmp"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
rows = image[120, 150:200]
x = np.arange(150, 200)
plt.figure(figsize=(7, 5))
plt.stem(x, rows)
plt.xlabel('Columns')
plt.ylabel('Pixel Value')
plt.title('Pixel Values for Row 120 and Columns 150:200')
plt.grid(True)
plt.show()


# Q.1) b) 

elvis_image = cv2.imread(image_path)
kernel = np.array([
[0,-1/4, 0],
[-1/4, 2,-1/4],
[0,-1/4, 0]])
convolved_image = cv2.filter2D(elvis_image,-1, kernel)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(elvis_image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(convolved_image)
plt.title('Convolved Image')
plt.axis('off')
plt.show()


# Q.1) c)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
rows = image[120, 150:200]
x = np.arange(150, 200)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.stem(x, rows)
plt.title('Original Image Row Pixel Values')
plt.xlabel('Column Index')
plt.ylabel('Pixel Value')
plt.grid(True)
kernel = np.array([[0,-1/4, 0],
[-1/4, 2,-1/4],
[0,-1/4, 0]])
convolved_image = cv2.filter2D(image,-1, kernel)
rows2 = convolved_image[120, 150:200]
x2 = np.arange(150, 200)
plt.subplot(122)
plt.stem(x2, rows2)
plt.title('Convolved Image Row Pixel Values')
plt.xlabel('Column Index')
plt.ylabel('Pixel Value')
plt.grid(True)
plt.show()

# Q.1) d)

image = cv2.imread(image_path)
kernel = np.array([[0,-1/4, 0],
[-1/4, 2,-1/4],
[0,-1/4, 0]])
convolved_image = cv2.filter2D(image,-1, kernel)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_convolved_image = cv2.cvtColor(convolved_image, cv2.COLOR_BGR2GRAY)
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_convolved = cv2.calcHist([gray_convolved_image], [0], None, [256], [0, 256])
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(hist_original)
plt.title("Histogram of Original Image")
plt.subplot(122)
plt.plot(hist_convolved)
plt.title("Histogram of Convolved Image")
plt.show()