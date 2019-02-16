# -*- coding: utf-8 -*-

import keras
import numpy
import random

from utils.label_utils import get_label_props
from utils.image_utils import get_image_props, IMAGE_HEIGHT, IMAGE_WIDTH
from utils.props_utils import *


def create_model(model_path=None, continue_train=False):
    if model_path is not None and continue_train:
        model = keras.models.load_model(filepath=model_path)
    else:
        model = keras.Sequential()

        model.add(keras.layers.Conv2D(input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                                      filters=512, kernel_size=(5, 5), padding='SAME', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.1))

        model.add(keras.layers.Dense(units=32, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.1))

        model.add(keras.layers.Dense(units=2, activation='sigmoid'))

        # 注意当前数据集太少了，如果使用0.01的学习速率则5个epoch之内就无法训练了。
        # （这个情况出现在gpu服务器上，本机目前没发现这个问题。）
        model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
                      loss='binary_crossentropy',
                      metrics=['acc'])

    return model


def run(train_images_dir, train_labels_path, model_path=None, continue_train=False):
    # current format:
    #   image_props: {id:image,...}
    #   label_props: {id:label,...}
    # -->
    # target format:
    #   x: [image1,image2,...]
    #   y: [label1,label2,...]
    label_props, label_titles = get_label_props(labels_path=train_labels_path)
    image_props, image_titles = get_image_props(rgb_image_dir=train_images_dir)

    # {id1:[props1,props2,...],...}
    # {id1:[label,image],...}
    all_props = merge_all_props(label_props, [image_props])
    all_titles = merge_all_titles(label_titles, [image_titles])

    # print(all_titles)
    # print(all_props)

    all_props_value = list(all_props.values())
    # [[label,image],[],[],...]
    # 打乱数据（当前net数据集是完全有序的）
    random.shuffle(all_props_value)

    all_props_value = list(zip(*all_props_value))
    # print(all_props_value)

    x = numpy.array(list(all_props_value[1]))
    y = numpy.array(list(all_props_value[0]))

    train_cnt = int(0.7 * x.shape[0])
    x_train = x[:train_cnt]
    y_train = y[:train_cnt]
    x_test = x[train_cnt:]
    y_test = y[train_cnt:]

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    print('x_train.shape=', x_train.shape)
    print('y_train.shape=', y_train.shape)

    # x [样本数量，height，width，channels]
    # x [num_samples，height，width，channels]
    # y [样本数量，label]
    # y [num_samples，label]

    model = create_model(model_path, continue_train)

    # history = model.fit(x, y, batch_size=10, epochs=10000, verbose=1, validation_split=0.2)
    # 由于gpu服务器训练很快就达到了100%的验证准确率。
    # 可能存在由于数据集太小的问题，此处将验证集设置为全部数据集（一定程度让验证集的结果值更加有意义）。
    if model_path is not None:
        import os.path

        if not os.path.exists(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        history = model.fit(x_train, y_train, batch_size=10, epochs=100, verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[keras.callbacks.ModelCheckpoint(filepath=model_path)])
    else:
        history = model.fit(x_train, y_train, batch_size=10, epochs=100, verbose=1,
                            validation_data=(x_test, y_test))

    eval_result = model.evaluate(x_test, y_test, batch_size=10, verbose=0)

    print('evaluate result:')
    print(eval_result)


if __name__ == '__main__':
    train_images_dir = './../../resources/input/net/images'
    train_labels_path = './../../resources/input/net/labels/labels.csv'
    run(train_images_dir=train_images_dir, train_labels_path=train_labels_path,
        model_path='./../../resources/models/cnn_classifier.h5')
