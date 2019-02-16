# -*- coding: utf-8 -*-

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

if __name__ == '__main__':
    all_props_path = './../../resources/output/net/all_props.csv'

    adaboost_classifier.run(all_props_path=all_props_path)
    bagging_classifier.run(all_props_path=all_props_path)
    random_forest_classifier.run(all_props_path=all_props_path)

    bernoulli_nb_classifier.run(all_props_path=all_props_path)
    gaussian_nb_classifier.run(all_props_path=all_props_path)
    multinomial_nb_classifier.run(all_props_path=all_props_path)
    decision_tree_classifier.run(all_props_path=all_props_path)
    knn_classifier.run(all_props_path=all_props_path)
    svm_classifier.run(all_props_path=all_props_path)
    voting_classifier.run(all_props_path=all_props_path)
