import numpy
import csv
from utils.glcm_utils import calc_glcm_properties


# all_props is a list.
# all_props[i] is a kind of features.
# all_props[i] is a dict, key is id, value is feature value.
def merge_all_props(labels, all_props):
    merged_props = {}
    for id in labels:
        merged_props[id] = []
        merged_props[id].extend(labels[id])
        for props in all_props:
            merged_props[id].extend(props[id])
    return merged_props


def get_label_props(labels_path):
    label_props = {}
    with open(labels_path, 'rt') as labels_file:
        csv_reader = csv.reader(labels_file, delimiter=',')
        csv_reader = list(csv_reader)
        for row in csv_reader[1:]:
            label_props[row[0]] = row[1:]
    return label_props


def output2csv(props, titles, filepath):
    with open(filepath, mode='wt', newline='') as file:
        csv_writer = csv.writer(file, delimiter=',')
        csv_writer.writerow(["id", *(titles)])
        for id in props:
            csv_writer.writerow([id, *(props[id])])


# 定义程序的步骤。
def start():
    # 1: 提取图片的纹理特征。
    glcm_props, glcm_prop_titles = calc_glcm_properties(rgb_image_dir='./../resources/input/net/images',
                                                        output_dir='./../resources/output/net',
                                                        distances=[1, 2],
                                                        angles=[0, numpy.pi / 4.0])
    output2csv(glcm_props,glcm_prop_titles,'./../resources/output/net/glcm_props.csv')
    # 2:
    # 3:
    # 4: 获取labels属性。
    label_props = get_label_props(labels_path='./../resources/input/net/labels/labels.csv')
    # 5: 合并所有特征。
    all_props = merge_all_props(labels=label_props, all_props=[glcm_props])
    print(all_props)


if __name__ == '__main__':
    start()