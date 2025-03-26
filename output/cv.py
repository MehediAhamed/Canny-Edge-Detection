import cv2
import numpy as np
import matplotlib.pyplot as plt

def non_maximum_suppression(gradient_magnitude, gradient_direction):

    M, N = gradient_magnitude.shape
    suppressed_img = np.zeros((M, N), dtype=np.float32)

    angle = gradient_direction / 180.0 * np.pi  # Convert angle to radians
    angle = np.round(angle / (np.pi / 4)) * (np.pi / 4)  # Quantize angles to 4 directions (0, 45, 90, 135)

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

def main():
    # Load the image
    image_filename = 'original.jpg'  # Make sure to update the path accordingly
    img_color = cv2.imread(image_filename) # Original color image
    
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    img_gray = cv2.imread(image_filename, 0)  # Grayscale image

    if img_gray is None:
        print('Could not open or find the image:', image_filename)
        exit(0)

    # Step 1: Apply Gaussian filter to remove noise (This is done automatically in OpenCV's Canny function)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)

    # Step 2: Apply Sobel filters to get gradient magnitude and directions
    sobel_x = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_direction = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

    # Step 3: Apply Non-Maximum Suppression
    suppressed_img = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Step 4: Apply Double Thresholding and Hysteresis (This is also handled by OpenCV's Canny function)
    low_threshold = 50
    high_threshold = 190
    edges = cv2.Canny(img_blurred, low_threshold, high_threshold)

    # Create thresholded images for visual comparison
    low_threshold_img = np.zeros_like(img_gray)
    high_threshold_img = np.zeros_like(img_gray)
    low_threshold_img[img_gray >= low_threshold] = 255
    high_threshold_img[img_gray >= high_threshold] = 255

    # Display the results
    plt.figure(figsize=(8, 16))

    # List of images and titles to display
    images = [img_color, img_gray, img_blurred, np.uint8(gradient_magnitude), 
              np.uint8(gradient_direction), np.uint8(suppressed_img), edges,
              low_threshold_img, high_threshold_img]
    titles = ['Original Image Color', 'Grayscale Image', 'Blurred Image (Gaussian)', 
              'Gradient Magnitude Image', 'Gradient Direction Image', 'Non-Maximum Suppressed Image', 'Low Thresholded Image', 'High Thresholded Image',
              'Canny Edge Detected Image']

    # Create subplots to display multiple images
    for i in range(len(images)):
        plt.subplot(3, 4, i+1)  # Arrange images in a 3x4 grid
        plt.imshow(images[i], cmap='gray' if i != 0 else None)  # 'gray' for grayscale images
        plt.title(titles[i])
        plt.axis('off')  # Hide axes
        plt.subplots_adjust(wspace=0.5, hspace=0.2)  # Adjust space between images
        print("\n\n")
    plt.tight_layout(pad=3.0)  # Add padding between images for better clarity
    plt.show()

    # Display the results and save images
    cv2.imshow('Original Image Color', img_color)
    cv2.imwrite('original.jpg', img_color)

    cv2.imshow('Grayscale Image', img_gray)
    cv2.imwrite('gray.jpg', img_gray)

    cv2.imshow('Blurred Image (Gaussian)', img_blurred)
    cv2.imwrite('gaussian.jpg', img_blurred)

    cv2.imshow('Gradient Magnitude Image', np.uint8(gradient_magnitude))
    cv2.imwrite('magnitude.jpg', np.uint8(gradient_magnitude))

    cv2.imshow('Gradient Direction Image', np.uint8(gradient_direction))  
    cv2.imwrite('gradient.jpg', np.uint8(gradient_direction))

    cv2.imshow('Supressed Image', np.uint8(suppressed_img))  
    cv2.imwrite('non-maxima.jpg', np.uint8(suppressed_img))

    cv2.imshow('Low Thresholded Image (50)', low_threshold_img)
    cv2.imwrite('low_threshold.jpg', low_threshold_img)

    cv2.imshow('High Thresholded Image (190)', high_threshold_img)
    cv2.imwrite('high_threshold.jpg', high_threshold_img)


    cv2.imshow('Canny Edge Detected Image', edges)
    cv2.imwrite('canny.jpg', edges)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
