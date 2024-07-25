import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import convolve2d
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


# Q.2) a)

import numpy as np
import matplotlib.pyplot as plt
def zone_plate(rows, columns, f):
    x, y = np.meshgrid(np.linspace(-1, 1, columns), np.linspace(-1, 1, rows))
    zone_plate = 0.5 + 0.5 * np.cos(np.pi * f * (x**2 + y**2))
    return zone_plate
def display_image(image, title):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.show()
rows = int(input("Enter the number of rows: "))
columns = int(input("Enter the number of columns: "))
f = float(input("Enter the value of 'f': "))
output = zone_plate(rows, columns, f)
title = f"Zone Plate Image with f = {int(f)}"
display_image(output, title)


# Q.2) b)

plt.rcParams['animation.ffmpeg_path'] = '' #Put the path where ffmpeg is downloaded
NROWS, NCOLS=240, 320
def zone_plate(f, Nx, Ny):
    x, y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    zone_plate = 0.5 + 0.5 * np.cos(np.pi * f * (x**2 + y**2))
    return zone_plate
fig = plt.figure()
f = 1
def animate(i):
    image = zone_plate(f+1.0*i, NCOLS, NROWS)
    plt.imshow(image, cmap='gray')
ani = animation.FuncAnimation(fig, animate, frames=50)
FFwriter = animation.FFMpegWriter(codec='rawvideo')
ani.save('zoneplate.mov', writer=FFwriter)

# Q.2) c)

plt.rcParams['animation.ffmpeg_path'] = '' #Put the path where ffmpeg is downloaded
NROWS, NCOLS=240, 320
def zone_plate(f, Nx, Ny):
    x, y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    zone_plate = 0.5 + 0.5 * np.sin(np.pi * f * (x**2 + y**2))
    return zone_plate
def lpf(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    return convolve2d(image, kernel, mode='same', boundary='wrap')
f = 1
original_frame = zone_plate(f+49.0, NCOLS, NROWS)
filtered_frame = lpf(original_frame, kernel_size=2)
plt.subplot(1, 2, 1)
plt.imshow(original_frame, cmap='gray')
plt.title('Original Final Frame')
plt.subplot(1, 2, 2)
plt.imshow(filtered_frame, cmap='gray')
plt.title('Filtered Final Frame')
plt.show()