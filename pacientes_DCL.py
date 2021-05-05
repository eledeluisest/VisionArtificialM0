# Python
import numpy as np
import cv2
from cv2 import xfeatures2d
import PythonSIFT.pysift as pysift
from matplotlib import pyplot as plt
import logging
import time
from skimage import feature
import os
import pandas as pd

list_fichero_sanos = os.listdir('DibujosMuestra/dataset3_DCL/dibujos/sanos')
list_fichero_enfermos_est = os.listdir('DibujosMuestra/dataset3_DCL/dibujos/enfermos_DCL_estables')
list_fichero_enfermos_ev = os.listdir('DibujosMuestra/dataset3_DCL/dibujos/enfermos_DCL_evolucion')

list_enfermos = list_fichero_enfermos_est + list_fichero_enfermos_ev

# Vamos a leer y generar detectores para el modelo en primer lugar.
list_modelos = os.listdir('DibujosMuestra/dataset3_DCL/modelos')
modelos = {}
list_patrones = []
for modelo in list_modelos:
    imagen = cv2.imread('DibujosMuestra/dataset3_DCL/modelos/' + modelo,
                                           flags=cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1_sift, des_sift = sift.detectAndCompute(imagen, None)

    orb = cv2.ORB_create()
    kp1_orb, des_orb = orb.detectAndCompute(imagen, None)

    fast = cv2.FastFeatureDetector_create()
    kp1 = fast.detect(imagen, None)
    br = cv2.BRISK_create()
    kp_fast, des_fast = br.compute(imagen, kp1)
    list_patrones.append([modelo.split('.')[0], des_sift, des_orb, des_fast])

list_imgs = []
list_nombre_modelos = [x.split('.')[0] for x in list_modelos]
list_registros = []
for id_paciente, fichero in enumerate(list_fichero_sanos):
    if fichero != '.DS_Store':
        list_imgs = os.listdir('DibujosMuestra/dataset3_DCL/dibujos/sanos/' + fichero)
        for img in list_imgs:
            imagen = cv2.imread('DibujosMuestra/dataset3_DCL/dibujos/sanos/' + fichero + '/' + img,
                                 flags=cv2.IMREAD_GRAYSCALE)
            patron = [x for x in list_nombre_modelos if x in img]
            sift = cv2.SIFT_create()
            kp1_sift, des1_sift = sift.detectAndCompute(imagen, None)
            orb = cv2.ORB_create()
            kp1_orb, des1_orb = orb.detectAndCompute(imagen, None)
            fast = cv2.FastFeatureDetector_create()
            kp1 = fast.detect(imagen, None)
            br = cv2.BRISK_create()
            kp_fast, des_fast = br.compute(imagen, kp1)
            if len(patron) > 0:
                list_registros.append([fichero[2:], 0, patron[0], des1_sift, des1_orb, des_fast])


for id_paciente, fichero in enumerate(list_fichero_enfermos_est):
    if fichero != '.DS_Store':
        list_imgs = os.listdir('DibujosMuestra/dataset3_DCL/dibujos/enfermos_DCL_estables/' + fichero)
        for img in list_imgs:
            imagen = cv2.imread('DibujosMuestra/dataset3_DCL/dibujos/enfermos_DCL_estables/' + fichero + '/' + img,
                                flags=cv2.IMREAD_GRAYSCALE)
            patron = [x for x in list_nombre_modelos if x in img]
            sift = cv2.SIFT_create()
            kp1_sift, des1_sift = sift.detectAndCompute(imagen, None)
            orb = cv2.ORB_create()
            kp1_orb, des1_orb = orb.detectAndCompute(imagen, None)
            fast = cv2.FastFeatureDetector_create()
            kp1 = fast.detect(imagen, None)
            br = cv2.BRISK_create()
            kp_fast, des_fast = br.compute(imagen, kp1)
            if len(patron) > 0:
                list_registros.append([fichero[2:], 1, patron[0], des1_sift, des1_orb, des_fast])



for id_paciente, fichero in enumerate(list_fichero_enfermos_ev):
    if fichero != '.DS_Store':
        list_imgs = os.listdir('DibujosMuestra/dataset3_DCL/dibujos/enfermos_DCL_evolucion/' + fichero)
        for img in list_imgs:
            imagen = cv2.imread('DibujosMuestra/dataset3_DCL/dibujos/enfermos_DCL_evolucion/' + fichero + '/' + img,
                                flags=cv2.IMREAD_GRAYSCALE)
            patron = [x for x in list_nombre_modelos if x in img]
            sift = cv2.SIFT_create()
            kp1_sift, des1_sift = sift.detectAndCompute(imagen, None)
            orb = cv2.ORB_create()
            kp1_orb, des1_orb = orb.detectAndCompute(imagen, None)
            fast = cv2.FastFeatureDetector_create()
            kp1 = fast.detect(imagen, None)
            br = cv2.BRISK_create()
            kp_fast, des_fast = br.compute(imagen, kp1)
            if len(patron) > 0:
                print('Entro')
                list_registros.append([fichero[2:], 1, patron[0], des1_sift, des1_orb, des_fast])


df_patrones = pd.DataFrame(list_patrones, columns=['patron_reg', 'sift_pat', 'orb_pat', 'fast_pat'])
df_registros = pd.DataFrame(list_registros, columns=['id_paciente', 'target','patron_reg', 'sift_reg', 'orb_reg', 'fast_reg'])

df_all = pd.merge(df_registros, df_patrones, on=['patron_reg'], how='left')

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
df_all['fast_n_match'] = df_all.apply(lambda x: len(bf.match(x['fast_reg'], x['fast_pat'])), axis=1)


def match_orb(bf, col1, col2):
    try:
        return len(bf.match(col1, col2))
    except Exception:
        return 0

df_all['orb_n_match'] = df_all.apply(lambda x: match_orb(bf, x['orb_reg'], x['orb_pat']), axis=1)

# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def match_sift(flann, col1, col2):
    try:
        matches = flann.knnMatch(col1, col2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                if m not in good:
                    good.append(m)
        return len(good)
    except Exception:
        return 0

df_all['sift_n_match'] = df_all.apply(lambda x: match_sift(flann, x['sift_reg'], x['sift_pat']), axis=1)

df_all.loc[:,['id_paciente', 'target', 'patron_reg', 'orb_n_match', 'fast_n_match', 'sift_n_match']].to_csv('res/data_tab.csv')