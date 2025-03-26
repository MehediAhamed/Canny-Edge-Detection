import numpy as np
import math

def compute_gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size), dtype=float)
    for x in range(-kernel_size // 2 + 1, kernel_size // 2 + 1):
        for y in range(-kernel_size // 2 + 1, kernel_size // 2 + 1):
            kernel[x + kernel_size // 2, y + kernel_size // 2] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    denom = np.sum(kernel)
    kernel = kernel / denom  
    return kernel, denom  

def convolve(img: np.array, kernel: np.array) -> np.array:
    kernel_size = kernel.shape[0]
    output_size = (img.shape[0] - kernel_size + 1, img.shape[1] - kernel_size + 1)
    output_img = np.zeros((output_size[0], output_size[1]), dtype=img.dtype)
    
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            mat = img[i:i + kernel_size, j:j + kernel_size]
            output_img[i, j] = np.clip(np.sum(np.multiply(mat, kernel)), 0, 255)
    
    return output_img

def apply_sobel_filters(img):
    sobel_kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    img_x = convolve(img, sobel_kx)
    img_y = convolve(img, sobel_ky)

    G = np.hypot(img_x, img_y)
    G = G / G.max() * 255
    theta = np.arctan2(img_y, img_x)
    return G, theta



def non_maximal_suppression(gradient_magnitude, gradient_direction):

    M, N = gradient_magnitude.shape
    suppressed_img = np.zeros((M, N), dtype=np.float32)

    angle = gradient_direction / 180.0 * np.pi 
    angle = np.round(angle / (np.pi / 4)) * (np.pi / 4) 

    for i in range(1, M-1):
        for j in range(1, N-1):
            # Get the angle of the gradient direction at pixel (i, j)
            current_angle = angle[i, j]

            # Check the four possible gradient directions
            if (0 <= current_angle < np.pi / 8) or (7 * np.pi / 8 <= current_angle <= np.pi):
                # Horizontal direction (0 degrees)
                neighbor1 = gradient_magnitude[i, j-1]
                neighbor2 = gradient_magnitude[i, j+1]
            elif (np.pi / 8 <= current_angle < 3 * np.pi / 8):
                # 45 degrees
                neighbor1 = gradient_magnitude[i-1, j+1]
                neighbor2 = gradient_magnitude[i+1, j-1]
            elif (3 * np.pi / 8 <= current_angle < 5 * np.pi / 8):
                # Vertical direction (90 degrees)
                neighbor1 = gradient_magnitude[i-1, j]
                neighbor2 = gradient_magnitude[i+1, j]
            elif (5 * np.pi / 8 <= current_angle < 7 * np.pi / 8):
                # 135 degrees
                neighbor1 = gradient_magnitude[i-1, j-1]
                neighbor2 = gradient_magnitude[i+1, j+1]

            # Suppress the pixel if it's not the local maximum
            if (gradient_magnitude[i, j] >= neighbor1) and (gradient_magnitude[i, j] >= neighbor2):
                suppressed_img[i, j] = gradient_magnitude[i, j]
            else:
                suppressed_img[i, j] = 0

    return suppressed_img


def double_threshold_image(img, lowThreshold, highThreshold, weak_value):
    M, N = img.shape
    thresholded_img = np.zeros((M, N), dtype=np.int32)
    strong_value = 255
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    strong_i, strong_j = np.where(img > highThreshold)

    thresholded_img[strong_i, strong_j] = strong_value
    thresholded_img[weak_i, weak_j] = weak_value
    thresholded_img[zeros_i, zeros_j] = 0
    return thresholded_img

def hysteresis(img, weak, strong=255):
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if (img[i, j] == weak):
                if ((img[i - 1, j - 1] == strong) or (img[i, j - 1] == strong) or
                        (img[i + 1, j - 1] == strong) or (img[i - 1, j] == strong) or
                        (img[i + 1, j] == strong) or (img[i - 1, j + 1] == strong) or
                        (img[i, j + 1] == strong) or (img[i + 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img
