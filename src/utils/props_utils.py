# -*- coding: utf-8 -*-

import csv


# all_props is a list.
# all_props[i] is a kind of feature.
# all_props[i] is a dict, key is id, value is feature values.
def merge_all_props(labels, all_props):
    merged_props = {}
    for id in labels:
        merged_props[id] = []
        merged_props[id].extend(labels[id])
        for props in all_props:
            merged_props[id].extend(props[id])
    return merged_props


# all_prop_titles is a list.
# all_prop_titles[i] is titles of a kind of feature.
def merge_all_titles(label_titles, all_prop_titles):
    merged_prop_titles = []
    merged_prop_titles.extend(label_titles)
    for prop_titles in all_prop_titles:
        merged_prop_titles.extend(prop_titles)
    return merged_prop_titles


# props is a dict, key is id, value is feature values.
# titles is a list
def output2csv(props, titles, filepath):
    with open(filepath, mode='wt', newline='') as file:
        csv_writer = csv.writer(file, delimiter=',')
        csv_writer.writerow(["id", *(titles)])
        for id in props:
            csv_writer.writerow([id, *(props[id])])
