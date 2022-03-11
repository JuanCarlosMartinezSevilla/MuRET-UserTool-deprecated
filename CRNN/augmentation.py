import cv2
import random
import numpy as np


class ImageModificator:
    Contrast = 0
    FishEye = 1
    Rotation = 2
    EroDila = 3
    Pcontrast = 0.25

    def __init__(self, flags):
        self.flags = flags

    def apply(self, image):
        if self.Contrast in self.flags:
            image = self.__apply_contrast(image)
        if self.EroDila in self.flags:
            image = self.__apply_erosion_dilation(image)
        if self.Rotation in self.flags:
            image = self.__apply_rotation(image)

        return image

    @staticmethod
    def __apply_rotation(image):
        matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), random.uniform(-1 * 3, 3), 1.0)
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    @staticmethod
    def __apply_contrast(image):
        if random.random() > ImageModificator.Pcontrast:
            return image

        clahe = cv2.createCLAHE(1.0)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def __apply_erosion_dilation(image):
        n = random.randint(-1 * 4, 4)
        kernel = np.ones((abs(n), abs(n)), np.uint8)

        if n < 0:
            return cv2.erode(image, kernel, iterations=1)

        return cv2.dilate(image, kernel, iterations=1)
