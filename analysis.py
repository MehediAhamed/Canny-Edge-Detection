import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a sample image (replace this with your own image)
image = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image path

# Normalize image to 0-1 for better visualization in Matplotlib
image = image / 255.0

# Apply Canny edge detection
edges_canny = cv2.Canny((image * 255).astype(np.uint8), 100, 200)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobel_x, sobel_y)

# Apply Prewitt edge detection (similar to Sobel)
kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(image, -1, kernel_x)
prewitt_y = cv2.filter2D(image, -1, kernel_y)
edges_prewitt = cv2.magnitude(prewitt_x, prewitt_y)

# Apply Laplacian edge detection
edges_laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Apply Roberts edge detection (approximate gradient)
roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
roberts_x = cv2.filter2D(image, -1, roberts_kernel_x)
roberts_y = cv2.filter2D(image, -1, roberts_kernel_y)
edges_roberts = cv2.magnitude(roberts_x, roberts_y)

# Apply Marr-Hildreth edge detection (Laplacian of Gaussian)
# First, apply Gaussian smoothing
sigma = 1.0
gaussian_blurred = cv2.GaussianBlur(image, (5, 5), sigma)
# Then, apply the Laplacian to the blurred image
edges_marr_hildreth = cv2.Laplacian(gaussian_blurred, cv2.CV_64F)

# Set up the figure for displaying the results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Titles for each subplot
methods = ['Canny', 'Sobel', 'Prewitt', 'Laplacian', 'Roberts', 'Marr-Hildreth']

# Plot each edge-detection result
axes[0, 0].imshow(edges_canny, cmap='gray')
axes[0, 0].set_title('Canny')
axes[0, 0].axis('off')

axes[0, 1].imshow(edges_sobel, cmap='gray')
axes[0, 1].set_title('Sobel')
axes[0, 1].axis('off')

axes[0, 2].imshow(edges_prewitt, cmap='gray')
axes[0, 2].set_title('Prewitt')
axes[0, 2].axis('off')

axes[1, 0].imshow(edges_laplacian, cmap='gray')
axes[1, 0].set_title('Laplacian')
axes[1, 0].axis('off')

axes[1, 1].imshow(edges_roberts, cmap='gray')
axes[1, 1].set_title('Roberts')
axes[1, 1].axis('off')

axes[1, 2].imshow(edges_marr_hildreth, cmap='gray')
axes[1, 2].set_title('Marr-Hildreth')
axes[1, 2].axis('off')

# Show the figure
plt.tight_layout()
plt.show()
