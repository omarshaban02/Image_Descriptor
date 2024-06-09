import sys
import threading
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import pyqtgraph as pg
import numpy as np
import cv2
from PyQt5.uic import loadUiType
from Classes.Image import Image
from Classes.Features import Features
from Classes.Sift import SIFT
from Classes.Thread import WorkerThread

ui, _ = loadUiType('main.ui')


class ImageDescriptor(QMainWindow, ui):
    def __init__(self):
        super(ImageDescriptor, self).__init__()
        self.setupUi(self)

        self.time_taken = 0
        self.loaded_image = None
        self.gray_detected_img = None
        self.color_detected_img = None
        self.features = None
        self.loaded_image_SIFT = None
        self.loaded_image_SIFT_1 = None
        self.loaded_image_SIFT_2 = None
        self.sift = SIFT()

        self.plotwidget_set = [self.wgt_img_input, self.wgt_edge_color, self.wgt_edge_grey_2,
                               self.wgt_SIFT_input_1, self.wgt_SIFT_input_2, self.wgt_SIFT_output_SSD,
                               self.wgt_SIFT_output_NCC, self.wgt_SIFT_input, self.wgt_SIFT_output]

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_input, self.item_output_grey, self.item_output_color,
                               self.item_SIFT_input_1, self.item_SIFT_input_2,
                               self.item_SIFT_output_SSD, self.item_SIFT_output_NCC,
                               self.item_SIFT_input, self.item_SIFT_output,
                               ] = [pg.ImageItem() for _ in range(9)]

        self.init_application()

        self.sift_thread = None
        self.match_thread = None

        ############################### Connections ##################################################
        # Connect Openfile Action to its function
        self.actionOpen.triggered.connect(lambda: self.open_image(0))
        self.lambda_chkBox.clicked.connect(self.corner_detection)
        self.harris_th_slider.valueChanged.connect(self.corner_detection)
        self.lambda_th_slider.valueChanged.connect(self.corner_detection)

        self.btn_SIFT_open_1.clicked.connect(lambda: self.open_image(1))
        self.btn_SIFT_open_2.clicked.connect(lambda: self.open_image(2))
        self.btn_SIFT_match.clicked.connect(lambda: self.output_matches(10))

        self.btn_SIFT_open_3.clicked.connect(lambda: self.open_image(3))
        self.btn_SIFT_match_2.clicked.connect(self.apply_sift)

    ################################ Corner Detection Lambda Minus ###############################
    def corner_detection(self):
        if self.features is not None:
            if self.lambda_chkBox.isChecked():
                self.color_detected_img, self.gray_detected_img, self.time_taken = \
                    self.features.find_corners(th=(self.lambda_th_slider.value() / 100))
                self.lambda_lcdNumber.display(self.time_taken)
                self.lambda_th_lcdNumber.display(self.lambda_th_slider.value() / 100)
                self.display_image(self.item_output_grey, self.gray_detected_img)
                self.display_image(self.item_output_color, self.color_detected_img)

            elif self.harris_chkBox.isChecked():
                self.color_detected_img, self.gray_detected_img, self.time_taken = \
                    self.features.harris_corner_detection(threshold=(self.harris_th_slider.value() / 1000))
                self.harris_lcdNumber.display(self.time_taken)
                self.harris_th_lcdNumber.display(self.harris_th_slider.value() / 1000)
                self.display_image(self.item_output_grey, self.gray_detected_img)
                self.display_image(self.item_output_color, self.color_detected_img)
        else:
            pass

    ##############################################################################################                
    ############################## SIFT/Harris Functions #########################################

    def calc_NCC(self, descriptor1, descriptor2, mean_1, mean_2):
        """Calculates the Normalized Cross Correlation for two image descriptors

        Args:
            descriptor1 : descriptor of image 1
            descriptor2 : descriptor of image 2
            mean_1      : mean of descriptors of image 1
            mean_2      : mean of descriptors of image 2
        """

        rms_1 = np.sqrt(np.sum((descriptor1 - mean_1) ** 2))
        rms_2 = np.sqrt(np.sum((descriptor2 - mean_2) ** 2))

        ncc = np.sum((descriptor1 - mean_1) * (descriptor2 - mean_2)) / (rms_1 * rms_2 + 1e-9)
        return ncc

    def calc_SSD(self, descriptor1, descriptor2):
        """Calculates the Sum of Squared Difference between two image descriptors

        Args:
            descriptor1 (np.ndarray): Numpy Array of descriptors for image 1
            descriptor2 (np.ndarray): Numpy Array of descriptors for image 2

        Returns:
            float: the Sum of Squared Difference
        """
        ssd = np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))
        return ssd

    def match_descriptors_ssd(self, descriptors_1, descriptors_2):
        matches = []

        # Matches each feature in image 1 with all featurs in image 2 and appends the best match
        for i, descriptor1 in enumerate(descriptors_1):
            best_match_index = -1
            best_match_score = np.inf

            for j, descriptor2 in enumerate(descriptors_2):
                ssd_score = self.calc_SSD(descriptor1, descriptor2)
                if ssd_score < best_match_score:
                    best_match_index = j
                    best_match_score = ssd_score

            matches.append((i, best_match_index))

        # Sort the resulting matches ascending by SSD Score (lower is better)    
        matches = sorted(matches, key=lambda x: self.calc_SSD(descriptors_1[x[0]], descriptors_2[x[1]]))

        return matches

    def match_descriptors_ncc(self, descriptors_1, descriptors_2):

        matches = []
        mean_1 = np.mean(descriptors_1, axis=0)
        mean_2 = np.mean(descriptors_2, axis=0)

        for i, descriptor1 in enumerate(descriptors_1):
            best_match_index = -1
            best_match_score = -np.inf

            for j, descriptor2 in enumerate(descriptors_2):
                ncc_score = self.calc_NCC(descriptor1, descriptor2, mean_1, mean_2)
                if ncc_score > best_match_score:
                    best_match_index = j
                    best_match_score = ncc_score

            matches.append((i, best_match_index))

        # Sort the resulting matches Descending by NNC Score (Higher is better)
        matches = sorted(matches, key=lambda x: self.calc_NCC(descriptors_1[x[0]], descriptors_2[x[1]], mean_1, mean_2),
                         reverse=True)

        return matches


    def output_matches(self, N=10):
        """Displays matching results for SIFT

        Args:
            N (int, optional): Number of desired matches to be displayed. Defaults to 10.
        """
        match_thread = threading.Thread(target=self.display_match, args=(N,))
        match_thread.start()

    def display_match(self,N):

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        keypoints1, descriptors_1 = sift.detectAndCompute(self.loaded_image_SIFT_1, None)
        keypoints2, descriptors_2 = sift.detectAndCompute(self.loaded_image_SIFT_2, None)

        matching_result_SSD = self.match_descriptors_ssd(descriptors_1, descriptors_2)
        matching_result_NCC = self.match_descriptors_ncc(descriptors_1, descriptors_2)

        # Convert indices to DMatch Objects for OpenCV drawMatches
        matching_result_SSD = [cv2.DMatch(idx1, idx2, 0) for idx1, idx2 in matching_result_SSD]
        matching_result_NCC = [cv2.DMatch(idx1, idx2, 0) for idx1, idx2 in matching_result_NCC]

        matching_result_SSD_img = cv2.drawMatches(self.loaded_image_SIFT_1, keypoints1, self.loaded_image_SIFT_2,
                                                  keypoints2, matching_result_SSD[:N], None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matching_result_NCC_img = cv2.drawMatches(self.loaded_image_SIFT_1, keypoints1, self.loaded_image_SIFT_2,
                                                  keypoints2, matching_result_NCC[:N], None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        self.display_image(self.item_SIFT_output_SSD, cv2.rotate(matching_result_SSD_img, cv2.ROTATE_90_CLOCKWISE))
        self.display_image(self.item_SIFT_output_NCC, cv2.rotate(matching_result_NCC_img, cv2.ROTATE_90_CLOCKWISE))

    ###################################### SIFT #########################################
    def apply_sift(self):
        self.sift_thread = WorkerThread(self.loaded_image_SIFT)
        self.sift_thread.start()
        self.sift_thread.signals.get_keypoints_descriptors.connect(self.display_sift_image)

        # return refined_keypoints, discriptors

    def display_sift_image(self, keypoints, descriptors):
        output_image = self.sift.draw_keypoints(self.loaded_image_SIFT, keypoints)
        self.display_image(self.item_SIFT_output, cv2.rotate(output_image, cv2.ROTATE_90_CLOCKWISE))

    ################################# Misc Functions #############################################

    def open_image(self, target_image=0):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif *.jpeg)")
        file_dialog.setWindowTitle("Open Image File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.load_img_file(selected_file, target_image)

    def load_img_file(self, image_path, target_image=0):

        # Loads the image using imread, converts it to RGB, then rotates it 90 degrees clockwise

        if target_image == 0:
            image = cv2.rotate(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
            self.loaded_image = image
            self.img_obj = Image(self.loaded_image)
            self.gray_scale_image = self.img_obj.gray_scale_image
            self.features = Features(self.loaded_image)
            self.display_image(self.item_input, self.loaded_image)

        elif target_image == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.loaded_image_SIFT_1 = image
            self.display_image(self.item_SIFT_input_1, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

        elif target_image == 2:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.loaded_image_SIFT_2 = image
            self.display_image(self.item_SIFT_input_2, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

        elif target_image == 3:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.loaded_image_SIFT = image
            self.display_image(self.item_SIFT_input, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

        # reset when load a new image
        self.lambda_lcdNumber.display(0)
        self.harris_lcdNumber.display(0)
        self.corner_detection()

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(image)
        image_item.getViewBox().autoRange()

    def setup_plotwidgets(self):
        for plotwidget in self.findChildren(pg.PlotWidget):
            # Removes Axes and Padding from all plotwidgets intended to display an image
            plotwidget.showAxis('left', False)
            plotwidget.showAxis('bottom', False)
            plotwidget.setBackground((25, 30, 40))
            plotitem = plotwidget.getPlotItem()
            plotitem.getViewBox().setDefaultPadding(0)

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)

    def setup_checkboxes(self):
        for checkbox in [self.lambda_chkBox, self.harris_chkBox]:
            checkbox.clicked.connect(self.corner_detection)

    def init_application(self):
        self.setup_plotwidgets()
        self.setup_checkboxes()


app = QApplication(sys.argv)
win = ImageDescriptor()
win.show()
app.exec()
