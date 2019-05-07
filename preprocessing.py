import os
import threading

import cv2


def crop(image):
    pass


def save(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def preprocess(image_path, output_path):
    image = cv2.imread(image_path, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop(image)
    save(image, output_path)
