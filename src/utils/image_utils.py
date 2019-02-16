# -*- coding: utf-8 -*-

import cv2

import os
import os.path

# 图片大小
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 100


def get_image_props(rgb_image_dir):
    image_props = {}
    image_props_titles = ['image']

    rgb_image_names = os.listdir(rgb_image_dir)
    for rgb_image_name in rgb_image_names:
        id = rgb_image_name.split('.')[0]
        rgb_image_path = os.path.join(rgb_image_dir, rgb_image_name)
        rgb_image = cv2.imread(rgb_image_path, 1)
        rgb_image = cv2.resize(src=rgb_image, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT))
        image_props[id] = [rgb_image]

    return image_props, image_props_titles


if __name__ == '__main__':
    test_images_dir = './../../resources/input/test/images'
    images_props = get_image_props(rgb_image_dir=test_images_dir)
    print(images_props)
