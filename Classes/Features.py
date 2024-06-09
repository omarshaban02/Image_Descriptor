import numpy as np
import cv2
import time

class Features:
    def __init__(self, image):
        self.image = image
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.sobel_x = None
        self.sobel_y = None
        self.sobel_xy = None

    def manual_sobel(self, img):
        """
        Applies the Sobel operator to calculate image gradients in x and y directions.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            Tuple[numpy.ndarray]: Gradients in x and y directions.
        """
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)

        return sobelx, sobely, sobelxy

    def find_corners(self, th=0.1):
        """
        Detects corners in an image using the lambda minus algorithm.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            None
        """
        # Load the image
        start_time = time.time()
        img = self.image

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator for gradient calculation
        Ix, Iy, Ixy = self.manual_sobel(gray_img)

        # Structure tensor elements
        Gxx = Ix * Ix
        Gyy = Iy * Iy
        Gxy = Ixy

        # Calculate eigenvalues of the structure tensor
        eigenvalues = np.zeros_like(gray_img, dtype=np.float32)
        for y in range(gray_img.shape[0]):
            for x in range(gray_img.shape[1]):
                M = np.array([[Gxx[y, x], Gxy[y, x]], [Gxy[y, x], Gyy[y, x]]], dtype=np.float32)
                eigenvalues[y, x] = np.linalg.eigvals(M).min()

        # Apply thresholding
        # eigenvalues_max = eigenvalues.max()
        # eigenvalues_scaled = eigenvalues / eigenvalues_max
        # corners = np.where(eigenvalues > threshold)

        threshold = th * eigenvalues.max()
        corners = np.where(eigenvalues > threshold)
        end_time = time.time()
        time_taken = end_time - start_time
        # Draw circles at detected corners on the original color image
        img_color = img.copy()
        for i, j in zip(*corners):
            cv2.circle(img_color, (j, i), 1, (0, 255, 0), -1)  # Green color

        # Draw circles at detected corners on the grayscale image
        gray_img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for i, j in zip(*corners):
            cv2.circle(gray_img_color, (j, i), 1, (0, 255, 0), -1)  # Green color

        return img_color, gray_img_color, time_taken

    def harris_corner_detection(self, ksize=3, k=0.04, threshold=0.01):
        start_time = time.time()
        gray = self.gray_image.copy()

        # Calculates Gradient (first derivative) using Sobel derivatives
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        # Harris corner response
        Ixx = dx ** 2
        Iyy = dy ** 2
        Ixy = dx * dy

        # Convolution with Gaussian filter
        Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigmaX=2)
        Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigmaX=2)
        Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigmaX=2)

        # Harris response
        det = Sxx * Syy - Sxy ** 2
        trace = Sxx + Syy
        R = det - k * (trace ** 2)

        # Non-maximum suppression
        R_max = np.max(R)
        R_scaled = R / R_max
        corners = np.argwhere(R_scaled > threshold)

        end_time = time.time()
        time_taken = end_time - start_time

        color_image_with_corners, gray_image_with_corners = self.draw_corners(corners, (255, 0, 0))

        return color_image_with_corners, gray_image_with_corners, time_taken

    def draw_corners(self, corners, color):
        gray_image = self.gray_image.copy()
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        color_image = self.image.copy()

        for corner in corners:
            cv2.circle(gray_image, (corner[1], corner[0]), 1, color, 1)
            cv2.circle(color_image, (corner[1], corner[0]), 1, color, 1)

        return color_image, gray_image