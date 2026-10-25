import cv2
import matplotlib.pyplot as plt
img = cv2.imread("jommuFLOWER.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Colour Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.hist(gray.ravel(), 256, [0, 256], color="black")
plt.title("Histogram of Grayscale Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")


plt.subplot(2, 3, 5)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalized Image")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.axis("off")

plt.subplot(2, 3, 6)
plt.hist(equalized.ravel(), 256, [0, 256], color="black")
plt.title("Histogram of Equalized Image")

plt.show()
