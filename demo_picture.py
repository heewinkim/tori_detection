# -*- coding: utf-8 -*-


from src.tori_detection import ToriDetection
import cv2
from time import time

tori = ToriDetection()

img = cv2.imread('/Users/hian/Desktop/tori/tori_hair.jpeg')

tic = time()
is_tori = tori.predict(img)
toc = time()

print("{}ms".format((toc-tic)*1000))

if is_tori:
    print('이미지는 토리를 포함합니다.')
else:
    print('이미지에 토리가 없습니다.')
