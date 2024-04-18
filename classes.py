import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QObject


class Image:
    def __init__(self, image):
        self._original_img = image
        self._gray_scale_image = self.calculate_gray_scale_image(self.original_img)
        self._img_histogram = self.calculate_image_histogram()
        self.equalized_img, self.equalized_hist = self.equalize_image()
        self.normalized_img, self.normalized_hist = self.equalize_image(normalize=True)
        self._bgr_img_histograms = self.calculate_bgr_histogram_and_distribution()

    @property
    def original_img(self):
        return self._original_img

    @property
    def gray_scale_image(self):
        return self._gray_scale_image

    @property
    def bgr_img_histograms(self):
        return self._bgr_img_histograms

    @property
    def img_histogram(self):
        return self._img_histogram

    def calculate_gray_scale_image(self, img):
        # Extract the B, G, and R channels
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Convert to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
        gray_image = 0.299 * r + 0.587 * g + 0.114 * b

        # Convert to uint8 data type
        return gray_image.astype(np.uint8)

    def calculate_bgr_histogram_and_distribution(self):
        # Split the image into its three color channels: B, G, and R
        b, g, r = self.original_img[:, :, 0], self.original_img[:, :, 1], self.original_img[:, :, 2]

        # Calculate histograms for each color channel
        hist_b, _ = np.histogram(b.flatten(), bins=256, range=(0.0, 256.0))
        hist_g, _ = np.histogram(g.flatten(), bins=256, range=(0.0, 256.0))
        hist_r, _ = np.histogram(r.flatten(), bins=256, range=(0.0, 256.0))

        # Calculate grayscale histogram
        gray_image = np.dot(self.original_img[..., :3], [0.2989, 0.5870, 0.1140])
        hist_gray, _ = np.histogram(gray_image.flatten(), bins=256, range=(0.0, 256.0))

        return hist_b, hist_g, hist_r, hist_gray

    def calculate_image_histogram(self):
        hist, _ = np.histogram(self.original_img.flatten(), 256, (0, 256))
        return hist

    def calculate_image_histogram_cv(self):
        hist = cv2.calcHist([self.original_img], [0], None, [256], [0, 256])
        return hist.flatten()

    def equalize_image(self, normalize=False):
        hist, _ = np.histogram(self.original_img.flatten(), 256, (0, 256))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        sk = np.round(cdf_normalized * 255)
        equalized_image = sk[self.original_img]
        equalized_hist, _ = np.histogram(equalized_image.flatten(), 256, (0, 256))
        if normalize:
            normalized_image = equalized_image / 255.0
            normalized_hist, _ = np.histogram(normalized_image.flatten(), 256, (0, 1), density=True)
            return normalized_image, normalized_hist
        return equalized_image, equalized_hist

    def equalize_image_cv(self, normalize=False):
        equalized_image = cv2.equalizeHist(self.original_img)
        if normalize:
            equalized_image = equalized_image / 255.0
        return equalized_image

class Features:
    def __init__(self, image):
        self.image = image
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

        return  sobelx, sobely, sobelxy
         

    def find_corners(self):
        """
        Detects corners in an image using the lambda minus algorithm.

        Parameters:
            image_path (str): Path to the input image.

        Returns:
            None
        """
        # Load the image
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
        threshold = 0.1 * eigenvalues.max()  # You can adjust this threshold
        corners = np.where(eigenvalues > threshold)

        # Draw circles at detected corners on the original color image
        img_color = img.copy()
        for i, j in zip(*corners):
            cv2.circle(img_color, (j, i), 1, (0, 255, 0), -1)  # Green color

        # Draw circles at detected corners on the grayscale image
        gray_img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for i, j in zip(*corners):
            cv2.circle(gray_img_color, (j, i), 1, (0, 255, 0), -1)  # Green color

        return img_color, gray_img_color