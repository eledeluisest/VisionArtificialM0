# Python
import numpy as np
import cv2
from cv2 import xfeatures2d
import PythonSIFT.pysift as pysift
from matplotlib import pyplot as plt
import logging
import time
from skimage import feature
logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 5

# Vamos a leer y generar detectores para el modelo en primer lugar.
img1 = cv2.imread('DibujosMuestra/dataset1_reyDistorsiones/Modelo.png', flags=cv2.IMREAD_GRAYSCALE)

t0_img1 = time.time()
lit_metodos = []

list_metodos = ['pysift', 'cv_sift', 'fast', 'brief', 'hog', 'orb']
dict_kp_des = {}
for METODO in list_metodos:
    if METODO == 'pysift':
        kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
        dict_kp_des[METODO] = [kp1, des1]
    elif METODO == 'cv_sift':
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        dict_kp_des[METODO] = [kp1, des1]
    elif METODO == 'surf':
        surf = cv2.xfeatures2d.SURF_create(400)
        kp1, des1 = surf.detectAndCompute(img1, None)
        dict_kp_des[METODO] = [kp1, des1]
    elif METODO == 'fast':
        fast = cv2.FastFeatureDetector_create()
        kp1 = fast.detect(img1, None)
        dict_kp_des[METODO] = [kp1, ' ']
    elif METODO == 'brief':
        star = cv2.xfeatures2d.StarDetector_create()
        kp = star.detect(img1, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1, des1 = brief.compute(img1, kp)
        dict_kp_des[METODO] = [kp1, des1]
    elif METODO == 'hog':
        fd, hog_image = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        dict_kp_des[METODO] = [fd, hog_image]
    elif METODO == 'orb':
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        dict_kp_des[METODO] = [kp1, des1]
