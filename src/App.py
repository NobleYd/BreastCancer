# -*- coding: utf-8 -*-

# 配置开始（配置可复现，即配置各种随机数产生seed使其每次都产生相同序列的随机数。）
#
# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926


def init():
    import os
    import numpy
    import tensorflow
    import random
    from keras import backend as K

    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    numpy.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    random.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tensorflow.set_random_seed(1234)

    sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
    K.set_session(sess)


# 配置结束
# ----------------------------------------------------------------------------------#

# 定义程序的步骤。
def start_calc_glcm_properties(dataset_name):
    import numpy
    from utils.props_utils import merge_all_props, merge_all_titles, output2csv
    from utils.label_utils import get_label_props
    from utils.glcm_utils import calc_glcm_properties

    # 1: 提取图片的纹理特征。
    glcm_props, glcm_prop_titles = calc_glcm_properties(
        rgb_image_dir='./../resources/input/{}/images'.format(dataset_name),
        output_dir='./../resources/output/{}'.format(dataset_name),
        distances=[4],
        angles=[0, numpy.pi / 4.0, numpy.pi / 2.0, 3 * numpy.pi / 4.0])
    output2csv(glcm_props, glcm_prop_titles, './../resources/output/{}/glcm_props.csv'.format(dataset_name))
    # 4: 获取labels属性。
    label_props, label_titles = get_label_props(
        labels_path='./../resources/input/{}/labels/labels.csv'.format(dataset_name))
    # 5: 合并所有特征。
    all_props = merge_all_props(labels=label_props, all_props=[glcm_props])
    all_titles = merge_all_titles(label_titles=label_titles, all_prop_titles=[glcm_prop_titles])
    output2csv(all_props, all_titles, './../resources/output/{}/all_props.csv'.format(dataset_name))


import argparse

from classifier import adaboost_classifier
from classifier import bagging_classifier
from classifier import random_forest_classifier

from classifier import gaussian_nb_classifier
from classifier import bernoulli_nb_classifier
from classifier import multinomial_nb_classifier

from classifier import decision_tree_classifier
from classifier import knn_classifier
from classifier import svm_classifier

from classifier import voting_classifier

import matplotlib.pyplot as pyplt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/Library/Fonts/Songti.ttc', size=12)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='App',
        usage='python App.py --mode=[glcm|all|vote|cnn] [options]',
        description='App entrance for brease cancer classify.'
    )
    parser.add_argument('--mode', choices=['glcm', 'all', 'vote', 'glcm_all', 'glcm_vote', 'cnn'], help='mode.')
    parser.add_argument('-c', '--continue_train', action='store_true', help='continue train mode.')

    args = parser.parse_args()

    # 无错误之后才会执行到此处。
    init()

    dataset_name = 'net200'

    all_props_path = './../resources/output/{}/all_props.csv'.format(dataset_name)

    if str.startswith(args.mode, 'glcm_'):
        start_calc_glcm_properties(dataset_name)

        args.mode = args.mode.split('_')[1]

    if args.mode == 'glcm':
        start_calc_glcm_properties(dataset_name)
    elif args.mode == 'all':
        # adaboost_classifier.run(all_props_path=all_props_path,show=False)
        # bagging_classifier.run(all_props_path=all_props_path,show=False)
        # random_forest_classifier.run(all_props_path=all_props_path,show=False)

        gaussian_nb_classifier.run(all_props_path=all_props_path, show=False)
        # bernoulli_nb_classifier.run(all_props_path=all_props_path,show=False)
        # multinomial_nb_classifier.run(all_props_path=all_props_path,show=False)

        decision_tree_classifier.run(all_props_path=all_props_path, show=False)
        knn_classifier.run(all_props_path=all_props_path, show=False)
        svm_classifier.run(all_props_path=all_props_path, show=False)
        voting_classifier.run(all_props_path=all_props_path, show=False)

        pyplt.title('ROC曲线', fontproperties=font)

        pyplt.xlabel('假阳性', fontproperties=font)
        pyplt.ylabel('真阳性', fontproperties=font)
        pyplt.legend()
        pyplt.show()


    elif args.mode == 'vote':
        voting_classifier.run(all_props_path=all_props_path)
    elif args.mode == 'cnn':
        from classifier import cnn_classifier

        cnn_classifier.run(train_images_dir='./../resources/input/{}/images'.format(dataset_name),
                           train_labels_path='./../resources/input/{}/labels/labels.csv'.format(dataset_name),
                           model_path='./../resources/models/cnn_classifier.h5',
                           continue_train=args.continue_train)
