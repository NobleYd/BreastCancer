import skimage.io
import skimage.color
import skimage.feature

import numpy

import csv

import os
import os.path


def calc_glcm_properties(rgb_image_dir, output_dir, distances, angles):
    """
    :param rgb_image_dir: RGB格式图片数据集目录。
    :param output_dir: 结果输出目录。
    """

    if not os.path.exists(os.path.join(output_dir, 'gray')):
        os.makedirs(os.path.join(output_dir, 'gray'))

    if isinstance(distances, int):
        distances = [distances]

    if isinstance(angles, int):
        angles = [angles]

    basic_prop_titles = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']

    glcm_prop_titles = []
    for i in range(len(distances)):
        for j in range(len(angles)):
            glcm_prop_titles.extend(
                [basic_prop_title + '-' + str(i) + '-' + str(j) for basic_prop_title in basic_prop_titles])

    glcm_props = {}
    rgb_image_names = os.listdir(rgb_image_dir)
    for rgb_image_name in rgb_image_names:
        id = rgb_image_name.split('.')[0]
        rgb_image_path = os.path.join(rgb_image_dir, rgb_image_name)

        # convert rgb image to gray image, and save to output dir.
        rgb_image = skimage.io.imread(rgb_image_path)
        gray_image = skimage.color.rgb2gray(rgb_image)
        skimage.io.imsave(os.path.join(output_dir, 'gray', rgb_image_name), gray_image)

        # 生成灰度共生矩阵
        gray_image = gray_image.astype(numpy.uint8)
        p = skimage.feature.greycomatrix(image=gray_image,
                                         distances=distances,
                                         angles=angles,
                                         symmetric=True, normed=True)

        # 提取纹理特征属性
        # contrast = skimage.feature.greycoprops(P=p, prop='contrast')
        # dissimilarity = skimage.feature.greycoprops(P=p, prop='dissimilarity')
        # homogeneity = skimage.feature.greycoprops(P=p, prop='homogeneity')
        # ASM = skimage.feature.greycoprops(P=p, prop='ASM')
        # energy = skimage.feature.greycoprops(P=p, prop='energy')
        # correlation = skimage.feature.greycoprops(P=p, prop='correlation')

        # print('contrast:', '\r\n', contrast, '\r\n')
        # print('dissimilarity:', '\r\n', dissimilarity, '\r\n')
        # print('homogeneity:', '\r\n', homogeneity, '\r\n')
        # print('ASM:', '\r\n', ASM, '\r\n')
        # print('energy:', '\r\n', energy, '\r\n')
        # print('correlation:', '\r\n', correlation, '\r\n')

        glcm_props[id] = [None] * len(basic_prop_titles)

        for i in range(len(basic_prop_titles)):
            glcm_props[id][i] = skimage.feature.greycoprops(P=p, prop=basic_prop_titles[i])

        glcm_props[id] = numpy.array(glcm_props[id])
        glcm_props[id] = glcm_props[id].transpose([1, 2, 0])
        glcm_props[id] = glcm_props[id].reshape([-1])

    return glcm_props, glcm_prop_titles


if __name__ == '__main__':
    testset_dir = './../../resources/input/testset'
    glcm_props = calc_glcm_properties(rgb_image_dir=testset_dir,
                                      output_dir='./../../resources/output/testset',
                                      distances=[1, 2],
                                      angles=[0, numpy.pi / 4.0])
    print(glcm_props)
