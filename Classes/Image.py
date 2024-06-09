import numpy as np
import cv2
import time
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
