import cv2
import numpy as np
import os
import utils as Utils  # Assuming 'utils.py' contains the necessary functions

def main():
    image_filename = 'flower.jpg'
    img_color = cv2.imread(image_filename)  # original color image
    img = cv2.imread(image_filename, 0)  # 0 indicates to read as grayscale

    if img is None:
        print('Could not open or find the image:', image_filename)
        exit(0)

    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Apply Gaussian filter to remove noise
    g, denom = Utils.compute_gaussian_kernel(3, 1.0)  # kernel size = 3, sigma = 1
    kernel = g / denom
    noise_removed_img = Utils.convolve(img, kernel)

    # Step 2: Apply Sobel filters, get gradient magnitude and angles
    gradient_magnitude_img, gradient_angles = Utils.apply_sobel_filters(noise_removed_img)

    # Step 3: Apply non-maximal suppression
    nonmax_suppressed_img = Utils.non_maximal_suppression(gradient_magnitude_img, gradient_angles)

    # Step 4: Apply thresholding and hysteresis
    low_threshold = 50
    high_threshold = 150
    weak_value = 25
    img_thresholded = Utils.double_threshold_image(nonmax_suppressed_img, low_threshold, high_threshold, weak_value)
    hysteresis_img = Utils.hysteresis(img_thresholded, weak_value)

    # Save images in the output folder
    cv2.imwrite(os.path.join(output_dir, 'original.jpg'), img_color)
    cv2.imwrite(os.path.join(output_dir, 'gray.jpg'), img)
    cv2.imwrite(os.path.join(output_dir, 'gaussian.jpg'), noise_removed_img.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'magnitude.jpg'), gradient_magnitude_img.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'non-maxima.jpg'), nonmax_suppressed_img.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'low_threshold.jpg'), img_thresholded.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, 'high_threshold.jpg'), hysteresis_img.astype(np.uint8))

    # Display results
    cv2.imshow('Original Image Color', img_color)
    cv2.imshow('Original Image - Grayscale', img)
    cv2.imshow('Gaussian Filtered Image', noise_removed_img.astype(np.uint8))
    cv2.imshow('Gradient Magnitude Image', gradient_magnitude_img.astype(np.uint8))
    cv2.imshow('NonMax Suppressed Image', nonmax_suppressed_img.astype(np.uint8))
    cv2.imshow('Canny Edge Detected Image', hysteresis_img.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
