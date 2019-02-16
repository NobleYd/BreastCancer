# -*- coding: utf-8 -*-

import csv


def get_label_props(labels_path):
    label_props = {}
    label_props_titles = []
    with open(labels_path, 'rt') as labels_file:
        csv_reader = csv.reader(labels_file, delimiter=',')
        rows = list(csv_reader)
        label_props_titles = rows[0][1:]
        for row in rows[1:]:
            label_props[row[0]] = row[1:]
    return label_props, label_props_titles


if __name__ == '__main__':
    test_labels_path = './../../resources/input/test/labels/labels.csv'
    label_props = get_label_props(labels_path=test_labels_path)
    print(label_props)
