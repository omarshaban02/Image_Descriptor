import numpy as np
import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal, QObject


class WorkerSignals(QObject):
    finished = pyqtSignal()
    update = pyqtSignal(np.ndarray)
    calc_area_perimeter = pyqtSignal(float, float)


class WorkerThread(QThread):
    def __init__(self, input_image, contour_points, epochs=100, edges_img=None):
        super(WorkerThread, self).__init__()
        self.signals = WorkerSignals()
        self.input_image = input_image
        self.contour_points = np.array(contour_points).astype(int)
        self.epochs = epochs
        self.edges_img = edges_img
        self.contour = None

    def run(self):
        pass


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


class SIFT:
    def __init__(self, num_octaves=3, num_scales=4, sigma=1.6, contrast_threshold=0.04, edge_threshold=10):
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold

    def generate_octave(self, image):
        octave = []
        for i in range(self.num_scales):
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=((2 ** 0.5) ** i) * self.sigma)
            octave.append(image)
        return np.array(octave)

    def generate_gaussian_pyramid(self, image):
        gaussian_pyramid = []
        scales = []

        for _ in range(self.num_octaves):
            octave_scales = []
            octave = self.generate_octave(image)
            gaussian_pyramid.append(octave)
            octave_scales.append(self.sigma)

            for _ in range(1, self.num_scales):
                self.sigma *= 2 ** (1 / self.num_scales)
                octave_scales.append(self.sigma)

            scales.append(octave_scales)

            image = cv2.resize(
                image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

        return gaussian_pyramid, scales

    def generate_DoG_pyramid(self, gaussian_pyramid):
        DoG_pyramid = []
        for octave in gaussian_pyramid:
            DoG_octave = [octave[i + 1] - octave[i] for i in range(len(octave) - 1)]
            DoG_pyramid.append(DoG_octave)
        return DoG_pyramid

    def scale_space_constuction(self, image):
        gaussian_pyramid, scales = self.generate_gaussian_pyramid(image)
        DoG_pyramid = self.generate_DoG_pyramid(gaussian_pyramid)
        return DoG_pyramid, scales

    def find_keypoints(self, DoG_pyramid):
        keypoints = []
        for octave_index, octave in enumerate(DoG_pyramid):
            for scale_index in range(1, len(octave) - 1):
                current_scale = octave[scale_index]
                lower_scale = octave[scale_index - 1]
                upper_scale = octave[scale_index + 1]

                # Define the neighborhood indices
                neighborhood_indices = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 0), (0, 1),
                    (1, -1), (1, 0), (1, 1)
                ]

                maxima_count = 0
                minima_count = 0
                for i in range(1, current_scale.shape[0] - 1):
                    for j in range(1, current_scale.shape[1] - 1):
                        pixel = current_scale[i, j]
                        is_maxima = True
                        is_minima = True
                        for di, dj in neighborhood_indices:
                            if not (
                                    lower_scale[i + di, j + dj] < pixel and upper_scale[i + di, j + dj] < pixel
                            ):
                                is_maxima = False
                            if not (
                                    lower_scale[i + di, j + dj] > pixel and upper_scale[i + di, j + dj] > pixel
                            ):
                                is_minima = False

                        if is_maxima:
                            maxima_count += 1
                            keypoints.append((i, j, octave_index))
                        elif is_minima:
                            minima_count += 1
                            keypoints.append((i, j, octave_index))

                print(f"Octave{octave_index}, Scale{scale_index}, No of maxima:", maxima_count)
                print(f"Octave{octave_index}, Scale{scale_index}, No of minima:", minima_count)
                print()

        print("Total keypoints:", len(keypoints))
        return keypoints

    def refine_keypoints(self, keypoints, DoG_pyramid, contrast_threshold=0.04, edge_threshold=10):
        refined_keypoints = []

        for i, j, octave_index in keypoints:
            current_octave = DoG_pyramid[octave_index]
            current_img = current_octave[1]

            # Check if the keypoint is within the image boundaries
            if 0 < i < current_img.shape[0] - 1 and 0 < j < current_img.shape[1] - 1:

                # Extract the current, upper, and lower scales
                current_scale = current_img[i, j]
                lower_scale = current_octave[0][i, j]
                upper_scale = current_octave[2][i, j]

                # Compute the difference of Gaussians
                delta_dog = np.abs(current_scale - lower_scale) + \
                            np.abs(current_scale - upper_scale)

                # Check if the keypoint is a local extremum
                if (current_scale > lower_scale and current_scale > upper_scale and
                        current_scale > contrast_threshold * delta_dog):

                    # Compute the curvature ratio
                    Dxx = current_img[i, j + 1] + \
                          current_img[i, j - 1] - 2 * current_scale
                    Dyy = current_img[i + 1, j] + \
                          current_img[i - 1, j] - 2 * current_scale
                    Dxy = (current_img[i + 1, j + 1] - current_img[i + 1, j - 1] -
                           current_img[i - 1, j + 1] + current_img[i - 1, j - 1]) / 4
                    trace = Dxx + Dyy
                    determinant = Dxx * Dyy - Dxy ** 2
                    curvature_ratio = trace ** 2 / \
                                      determinant if determinant != 0 else float('inf')

                    # Check if the curvature ratio satisfies the edge threshold
                    if curvature_ratio < (edge_threshold + 1) ** 2 / edge_threshold:
                        refined_keypoints.append((i, j, octave_index))

        print("Total refined_keypoints:", len(refined_keypoints))
        return refined_keypoints

    def assign_orientation(self, refined_keypoints, DoG_pyramid):
        oriented_keypoints = []

        for i, j, octave_index in refined_keypoints:
            current_octave = DoG_pyramid[octave_index]
            current_img = current_octave[1]

            window_size = 16
            i_min = max(0, i - window_size)
            i_max = min(current_img.shape[0], i + window_size)
            j_min = max(0, j - window_size)
            j_max = min(current_img.shape[1], j + window_size)
            window = current_img[i_min:i_max, j_min:j_max]

            magnitude, orientation = self.compute_gradients(window)

            orientation_bins = ((orientation * (180 / np.pi) + 180) / 10).astype(int)

            hist, _ = np.histogram(orientation_bins.flatten(), bins=range(37), weights=magnitude.flatten())

            max_bin = np.argmax(hist)
            max_orientation = max_bin * 10 - 180

            oriented_keypoints.append((i, j, max_orientation))

        return oriented_keypoints

    def compute_gradients(self, image):
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        orientation = np.arctan2(dy, dx)
        magnitude = magnitude.astype(np.float32)

        return magnitude, orientation

    def calculate_descriptor_vector(self, image, keypoints, window_size=16, num_bins=8):
        magnitude, orientation = self.compute_gradients(image)
        descriptor_vectors = []

        half_window = window_size // 2

        for keypoint in keypoints:
            keypoint_x, keypoint_y, _ = keypoint

            x_min = max(0, keypoint_x - half_window)
            x_max = min(magnitude.shape[0], keypoint_x + half_window)
            y_min = max(0, keypoint_y - half_window)
            y_max = min(magnitude.shape[1], keypoint_y + half_window)

            window_magnitude = magnitude[x_min:x_max, y_min:y_max]
            window_orientation = orientation[x_min:x_max, y_min:y_max]

            relative_orientation = window_orientation - \
                                   orientation[keypoint_x, keypoint_y]
            bin_index = np.clip(np.floor(
                (relative_orientation + 180) / (360 / num_bins)), 0, num_bins - 1).astype(int)

            histogram = np.zeros(num_bins)
            np.add.at(histogram, bin_index, window_magnitude)

            normalized_histogram = histogram / np.linalg.norm(histogram)
            descriptor_vectors.append(normalized_histogram)

        return descriptor_vectors

    def draw_keypoints_with_orientation(self, image, keypoints, orientations):
        output_image = np.copy(image)

        for keypoint, orientation in zip(keypoints, orientations):
            i, j, _ = keypoint
            cv2.circle(output_image, (j, i), radius=1, color=(255, 0, 0), thickness=1)
            angle = orientation * np.pi / 180
            endpoint_x = int(j + 10 * np.cos(angle))
            endpoint_y = int(i + 10 * np.sin(angle))
            cv2.line(output_image, (j, i), (endpoint_x, endpoint_y), color=(0, 255, 0), thickness=1)

        return output_image

    def draw_keypoints(self, image, keypoints):
        output_image = np.copy(image)

        for keypoint in keypoints:
            i, j, _ = keypoint
            cv2.circle(output_image, (j, i), radius=1, color=(255, 0, 0), thickness=-1)

        return output_image
