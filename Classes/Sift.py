import numpy as np
import cv2


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
