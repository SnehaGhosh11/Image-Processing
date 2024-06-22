import cv2
import numpy as np

# Load the image
image = cv2.imread('IMG147.jpg', cv2.IMREAD_GRAYSCALE)

# Subset the clear water and rock portion with dimension of (18x18)
subset = image[100:118, 100:118]

# Print the DN values from the subset image
print("DN values from the subset image:")
print(subset)

# Degrade image with addition of noise
noise = np.random.normal(0, 25, subset.shape)
noisy_subset = subset + noise.astype(np.uint8)

# Apply restoration method to restore image using suitable filtering techniques
# For example, applying Gaussian Blur for denoising
restored_image = cv2.GaussianBlur(noisy_subset, (5, 5), 0)

# Display the original subset, noisy subset, and restored image
cv2.imshow('Original Subset', subset)
cv2.imshow('Noisy Subset', noisy_subset)
cv2.imshow('Restored Image', restored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
