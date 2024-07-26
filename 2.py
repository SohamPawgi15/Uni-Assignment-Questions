
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import cv2

# Q.1) a)
# Low Frequency

im_size = 256
cycles = 8
x, y = np.meshgrid(np.linspace(0, 2 * np.pi, im_size), np.linspace(0, 2 * np.pi, im_size))
grating = np.sin(cycles * x)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grating, cmap='gray')
plt.axis('off')
plt.title(f'Sine Grating (Low Frequency)')
plt.subplot(1, 2, 2)
intensity = np.abs(grating[im_size // 2, :])
x = np.linspace(0, 17.5, im_size)
skip = 14
x = x[::skip]
intensity = intensity[::skip]
plt.stem(x, intensity, linefmt='b-', markerfmt='bo', basefmt='b-')
plt.xlabel('Column')
plt.ylabel('Intensity')
plt.title('Intensity Profile (Low Frequency)')
plt.grid(True)
plt.xlim(0, 17.5)
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(0, 18, 2.5))
plt.tight_layout()
plt.show()


# Medium Frequency

im_size = 256
cycles = 32
x, y = np.meshgrid(np.linspace(0, 2 * np.pi, im_size), np.linspace(0, 2 * np.pi, im_size))
grating = np.sin(cycles * x)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grating, cmap='gray')
plt.axis('off')
plt.title(f'Sine Grating (Medium Frequency)')
plt.subplot(1, 2, 2)
intensity = np.abs(grating[im_size // 2, :])
x = np.linspace(0, 17.5, im_size)
skip = 9
x = x[::skip]
intensity = intensity[::skip]
plt.stem(x, intensity, linefmt='b-', markerfmt='bo', basefmt='b-')
plt.xlabel('Column')
plt.ylabel('Intensity')
plt.title('Intensity Profile (Medium Frequency)')
plt.grid(True)
plt.xlim(0, 17.5)
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(0, 18, 2.5))
plt.tight_layout()
plt.show()


# Highest Frequency without aliasing

im_size = 256
cycles = 128
x, y = np.meshgrid(np.linspace(0, 2 * np.pi, im_size), np.linspace(0, 2 * np.pi, im_size))
grating = np.sin(cycles * x)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grating, cmap='gray')
plt.axis('off')
plt.title(f'Sine Grating (High Frequency without aliasing)')
plt.subplot(1, 2, 2)
intensity = np.abs(grating[im_size // 2, :])
x = np.linspace(0, 17.5, im_size)
skip = 8
x = x[::skip]
intensity = intensity[::skip]
plt.stem(x, intensity, linefmt='b-', markerfmt='bo', basefmt='b-')
plt.xlabel('Column')
plt.ylabel('Intensity')
plt.title('Intensity Profile (High Frequency)')
plt.grid(True)
plt.xlim(0, 17.5)
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(0, 18, 2.5))
plt.tight_layout()
plt.show()


# Q.1) b)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

size = 256
f_low = 8
f_medium = 32
f_high = 128
x = np.linspace(0, 1, size, endpoint=False)
y = np.linspace(0, 1, size, endpoint=False)
x_grid, y_grid = np.meshgrid(x, y)
g_low = np.sin(2 * np.pi * f_low * x_grid)
g_medium = np.sin(2 * np.pi * f_medium * x_grid)
g_high = np.sin(2 * np.pi * f_high * x_grid)
dft_low = fft2(g_low)
dft_medium = fft2(g_medium)
dft_high = fft2(g_high)
dft_low = fftshift(dft_low)
dft_medium = fftshift(dft_medium)
dft_high = fftshift(dft_high)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(np.abs(dft_low), cmap='gray')
plt.title('DFT Magnitude (Low Frequency)')
plt.subplot(132)
plt.imshow(np.abs(dft_medium), cmap='gray')
plt.title('DFT Magnitude (Medium Frequency)')
plt.subplot(133)
plt.imshow(np.abs(dft_high), cmap='gray')
plt.title('DFT Magnitude (Highest Frequency)')
plt.show()


# Q.2) a)

image_path = "images/venetianblindscrop.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
dft = fft2(image)
dft_shifted = fftshift(dft)
magnitude_spectrum = np.abs(dft_shifted)
plt.figure(figsize=(8, 10))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
plt.title('DFT Magnitude Spectrum')
plt.axis('off')
plt.show()

