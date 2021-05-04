
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
for i in range(1,8):
    img2 = cv2.imread('DibujosMuestra/dataset1_reyDistorsiones/t'+str(i)+'.png', flags=cv2.IMREAD_GRAYSCALE)
    t0_img1 = time.time()
    lit_metodos = []

    dict_kp_img = {}
    for nombre, imagen in [('modelo',img1), ('distorsion', img2)]:
        list_metodos = [ 'cv_sift', 'fast', 'orb']
        dict_kp_des = {}
        for METODO in list_metodos:
            if METODO == 'pysift':
                kp1, des1 = pysift.computeKeypointsAndDescriptors(imagen)
                dict_kp_des[METODO] = [kp1, des1]
            elif METODO == 'cv_sift':
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(imagen, None)
                dict_kp_des[METODO] = [kp1, des1]
            elif METODO == 'surf':
                surf = cv2.xfeatures2d.SURF_create(400)
                kp1, des1 = surf.detectAndCompute(imagen, None)
                dict_kp_des[METODO] = [kp1, des1]
            elif METODO == 'fast':
                fast = cv2.FastFeatureDetector_create()
                kp1 = fast.detect(imagen, None)
                br = cv2.BRISK_create()
                kp, des = br.compute(imagen, kp1)
                dict_kp_des[METODO] = [kp1, des]
            elif METODO == 'brief':
                star = cv2.xfeatures2d.StarDetector_create()
                kp = star.detect(imagen, None)
                brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                kp1, des1 = brief.compute(imagen, kp)
                dict_kp_des[METODO] = [kp1, des1]
            elif METODO == 'hog':
                fd, hog_image = feature.hog(imagen, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualize=True, multichannel=False)
                dict_kp_des[METODO] = [fd, hog_image]
            elif METODO == 'orb':
                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(imagen, None)
                dict_kp_des[METODO] = [kp1, des1]
        dict_kp_img[nombre] = dict_kp_des


    kp1, des1 = dict_kp_img['modelo']['orb']
    kp2, des2 = dict_kp_img['distorsion']['orb']

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.savefig('img_resultados/orb_d'+str(i)+'.png')
    plt.show()
    kp1, des1 = dict_kp_img['modelo']['fast']
    kp2, des2 = dict_kp_img['distorsion']['fast']

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.savefig('img_resultados/fast_d' + str(i) + '.png')
    plt.show()
    print(" Comienzo el test ")
    numero_d = i
    kp1, des1 = dict_kp_img['modelo']['cv_sift']
    kp2, des2 = dict_kp_img['distorsion']['cv_sift']
    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.35 * n.distance:
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
        print('img_resultados/sift_d' + str(numero_d) + '.png')
        plt.savefig('img_resultados/sift_d' + str(numero_d) + '.png')
        plt.show()

