"""
Comparación del algoritmo sift en Python basado en: https://github.com/rmislam/PythonSIFT
y el de OpenCV.

Se probará la eficiencia y los resultados obtenidos sobre tres pares de imágenes diferentes del entorno de Zaragoza.

"""


# Python
import numpy as np
import cv2
import PythonSIFT.pysift as pysift
from matplotlib import pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)

METODO = 'cv'
if METODO == 'cv':
    sift = cv2.SIFT_create()

MIN_MATCH_COUNT = 5

img1 = cv2.imread('img_comparativa/agua1.jpg', flags=cv2.IMREAD_GRAYSCALE)           # queryImage
img2 = cv2.imread('img_comparativa/agua2.jpg', 0)  # trainImage
bool_muestra = False
if bool_muestra:
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
# Compute SIFT keypoints and descriptors
print(" Descriptor sobre la imagen 1")
t0_img1 = time.time()
if METODO == 'pysift':
    kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
elif METODO == 'cv':
    kp1, des1 = sift.detectAndCompute(img1, None)
t1_img1 = time.time()
print("En procesar la imagen 1 he tardado:")
print((t1_img1-t0_img1)/60)

print(" Descriptor sobre la imagen 2")
t0_img2 = time.time()
if METODO == 'pysift':
    kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)
elif METODO == 'cv':
    kp2, des2 = sift.detectAndCompute(img2, None)
t1_img2 = time.time()
print("En procesar la imagen 2 he tardado:")
print((t1_img2-t0_img2)/60)

print(" Comienzo el test ")
# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        if m not in good:
            good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    if h2 > h1:
        hdif = int((h2 - h1) / 2)
    else:
        hdif = int((h1 - h2) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        if h2 > h1:
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2
        else:
            newimg[hdif:hdif + h2, :w2, i] = img2
            newimg[:h1, w2:w2 + w1, i] = img1

    # Draw SIFT keypoint matches
    for m in good:
        if h2 > h1:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))
        else:
            pt1 = (int(kp1[m.queryIdx].pt[0] + w2), int(kp1[m.queryIdx].pt[1]))
            pt2 = (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1] + hdif))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))


    plt.imshow(newimg)
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
