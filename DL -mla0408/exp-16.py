import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread(r"C:\Users\Prana\OneDrive\Desktop\dog.jpeg")

# Check if image was loaded successfully
if img is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Convert BGR to RGB
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Create kernel
    kernel = np.ones((2, 2), np.uint8)

    # Morphological closing
    closing = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Dilation
    sure_bg = cv2.dilate(
        closing,
        kernel,
        iterations=3
    )

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(232)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(233)
    plt.imshow(thresh, cmap="gray")
    plt.title("Otsu's Threshold")
    plt.axis("off")

    plt.subplot(234)
    plt.imshow(closing, cmap="gray")
    plt.title("MorphologyEx: Closing (2x2)")
    plt.axis("off")

    plt.subplot(235)
    plt.imshow(sure_bg, cmap="gray")
    plt.title("Dilation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save dilation result
    plt.imsave("dilation.png", sure_bg, cmap="gray")
