# -*- coding: utf-8 -*-
"""
===============================================
tori_detection module
===============================================

========== ====================================
========== ====================================
 Module     tori_detection module
 Date       2019-07-30
 Author     hian
 Comment    `관련문서링크 <call to heewinkim >`_
========== ====================================

*Abstract*
    * 토리 검출모듈

===============================================
"""

import os
import cv2
import numpy as np
from keras.models import load_model
from keras_applications.mobilenet_v2 import relu6
from keras.layers import DepthwiseConv2D
from keras.utils.generic_utils import CustomObjectScope


class ToriDetection(object):

    def __init__(self):

        current_dir = os.path.dirname(os.path.realpath(__file__))

        with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
            self._model = load_model(current_dir+'/tori.h5')
        self._input_size = (224,224)

        self._warm_up()

    def _warm_up(self):

        self._model.predict(np.ndarray((1,224, 224, 3), dtype=np.float32))
        self._model.predict(np.ndarray((1,224, 224, 3), dtype=np.float32))

    def _preprocessing(self,cv_img):
        img = cv2.resize(cv_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = np.expand_dims(img, 0)
        return np.array(img).astype('float32')

    def _predict(self, cv_img):
        """
        predict image include tori

        :param cv_img: cv_image(bgr)
        :param threshold: confidence threshold value
        :return: float,confidence
        """

        x = self._preprocessing(cv_img)
        prediction = self._model.predict(x)
        confidence = float(prediction[0][0])
        return confidence

    def predict(self,cv_img,threshold=0.75):
        """
        predict image include tori

        :param cv_img: cv_image(bgr)
        :param threshold: confidence threshold value
        :return: boolean
        """

        x = self._preprocessing(cv_img)
        prediction = self._model.predict(x)
        confidence = float(prediction[0][0])
        if confidence>=threshold:
            return True
        else:
            return False
