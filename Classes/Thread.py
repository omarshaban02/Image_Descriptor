import time
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from Classes.Sift import *


class WorkerSignals(QObject):
    get_keypoints_descriptors = pyqtSignal(list, list)


class WorkerThread(QThread):
    def __init__(self, input_image):
        super(WorkerThread, self).__init__()
        self.input_image = input_image
        self.signals = WorkerSignals()
        self.sift = SIFT()

    def run(self):
        t_start = time.time()
        DoG_pyramid, scales = self.sift.scale_space_constuction(self.input_image)
        keypoints = self.sift.find_keypoints(DoG_pyramid)
        refined_keypoints = self.sift.refine_keypoints(keypoints, DoG_pyramid)
        # orientations = self.sift.assign_orientation(refined_keypoints, DoG_pyramid)
        discriptors = self.sift.calculate_descriptor_vector(self.input_image, refined_keypoints)
        t_end = time.time()
        t_total = t_end - t_start
        print("SIFT is Done")
        print("Time taken in SIFT:", np.round(t_total, 2), "sec")
        self.signals.get_keypoints_descriptors.emit(refined_keypoints, discriptors)