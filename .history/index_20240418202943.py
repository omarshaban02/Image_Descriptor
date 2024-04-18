import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from ui import Ui_MainWindow
import pyqtgraph as pg
import numpy as np
import cv2
from PyQt5.uic import loadUiType
from classes import Image, Features
import time


ui, _ = loadUiType('main.ui')

class ImageDescriptor(QMainWindow, ui):
    def __init__(self):
        super(ImageDescriptor, self).__init__()
        self.time_taken = 0
        self.loaded_image = None
        self.gray_detected_img = None
        self.color_detected_img = None
        self.features = None
        self.setupUi(self)
        
        self.plotwidget_set = [self.wgt_img_input, self.wgt_edge_color, self.wgt_edge_grey]
        
        # Create an image item for each plot-widget
        self.image_item_set = [self.item_input, self.item_output_grey, self.item_output_color,
                               ] = [pg.ImageItem() for _ in range(3)]
        self.init_application()
        
        # Connect Openfile Action to its function
        self.actionOpen.triggered.connect(self.open_image)
        self.lambda_chkBox.clicked.connect(self.corner_detection)
    ############################### Connections ##################################################
            
    ################################ Corner Detection Lambda Minus ###############################
    def corner_detection(self):
        if self.features is not None:
            if self.lambda_chkBox.isChecked():
                self.color_detected_img, self.gray_detected_img, self.time_taken = self.features.find_corners()
                self.lambda_lcdNumber.display(self.time_taken)
                self.display_image(self.item_output_grey, self.gray_detected_img)
                self.display_image(self.item_output_color, self.color_detected_img)

            elif self.harris_chkBox.isChecked():
                self.color_detected_img, self.gray_detected_img, self.time_taken = self.features.harris_corner_detection()
                self.harris_lcdNumber.display(self.time_taken)
                self.display_image(self.item_output_grey,self.gray_detected_img)
                self.display_image(self.item_output_color, self.color_detected_img)
        else:
            pass
                
                
    ##############################################################################################                
    ############################## SIFT/Harris Functions #########################################
    
    def calc_NCC(descriptor1, descriptor2, mean_1, mean_2):
        """Calculates the Normalized Cross Correlation for two image descriptors

        Args:
            descriptor1 : descriptor of image 1
            descriptor2 : descriptor of image 2
            mean_1      : mean of descriptors of image 1
            mean_2      : mean of descriptors of image 2
        """

        rms_1 = np.sqrt(np.sum((descriptor1 - mean_1)**2))
        rms_2 = np.sqrt(np.sum((descriptor2 - mean_2)**2))

        ncc = np.sum((descriptor1 - mean_1) * (descriptor2 - mean_2)) / (rms_1 * rms_2 + 1e-9)
        return ncc
    
    
    def calc_SSD(descriptor1, descriptor2):
        ssd = np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))
        return ssd

    
    
    ##############################################################################################    
    ################################# Misc Functions #############################################
    
    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif *.jpeg)")
        file_dialog.setWindowTitle("Open Image File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.load_img_file(selected_file)
            
    def load_img_file(self, image_path):
        # Loads the image using imread, converts it to RGB, then rotates it 90 degrees clockwise
        self.loaded_image = cv2.rotate(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        self.img_obj = Image(self.loaded_image)
        self.gray_scale_image = self.img_obj.gray_scale_image
        self.features = Features(self.loaded_image)
        self.display_image(self.item_input, self.loaded_image)

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
