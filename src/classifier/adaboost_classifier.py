# -*- coding: utf-8 -*-

import matplotlib.pyplot as pyplt
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import csv
import numpy


def run(all_props_path, show=True):
    with open(all_props_path, 'rt') as all_props_file:
        csv_reader = csv.reader(all_props_file, delimiter=',')
        rows = numpy.array(list(csv_reader))

        titles = rows[0]
        all_props_value = rows[1:, 1:].astype(numpy.float32)

        numpy.random.shuffle(all_props_value)

        train_cnt = int(0.7 * all_props_value.shape[0])
        x_train = all_props_value[:train_cnt, 1:]
        y_train = all_props_value[:train_cnt, 0]
        x_test = all_props_value[train_cnt:, 1:]
        y_test = all_props_value[train_cnt:, 0]

        y_test = all_props_value[train_cnt:, 0]

        train_idx_0 = y_train == 0
        x_train_0 = x_train[train_idx_0]
        y_train_0 = y_train[train_idx_0]

        train_idx_1 = y_train == 1
        x_train_1 = x_train[train_idx_1]
        y_train_1 = y_train[train_idx_1]

        test_idx_0 = y_test == 0
        x_test_0 = x_test[test_idx_0]
        y_test_0 = y_test[test_idx_0]

        test_idx_1 = y_test == 1
        x_test_1 = x_test[test_idx_1]
        y_test_1 = y_test[test_idx_1]

        classifier = AdaBoostClassifier()

        classifier.fit(x_train, y_train)

        train_0_score = classifier.score(x_train_0, y_train_0)
        train_1_score = classifier.score(x_train_1, y_train_1)
        train_score = classifier.score(x_train, y_train)

        test_0_score = classifier.score(x_test_0, y_test_0)
        test_1_score = classifier.score(x_test_1, y_test_1)
        test_score = classifier.score(x_test, y_test)

        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, classifier.predict_proba(x_test)[:, 1], pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)

        print(__name__ + '.train.0.score=', train_0_score)
        print(__name__ + '.train.1.score=', train_1_score)
        print(__name__ + '.train.score=', train_score)

        print(__name__ + '.test.0.score= ', test_0_score)
        print(__name__ + '.test.1.score= ', test_1_score)
        print(__name__ + '.test.score= ', test_score)

        print(__name__ + '.auc        =', auc)

        print(__name__ + '.precision.0=',
              sklearn.metrics.precision_score(y_test, classifier.predict(x_test), pos_label=0))

        print(__name__ + '.recall.0   =',
              sklearn.metrics.recall_score(y_test, classifier.predict(x_test), pos_label=0))

        print(__name__ + '.precision.1=',
              sklearn.metrics.precision_score(y_test, classifier.predict(x_test), pos_label=1))

        print(__name__ + '.recall.1   =',
              sklearn.metrics.recall_score(y_test, classifier.predict(x_test), pos_label=1))

        pyplt.title('adaboost_classifier, train_score:{:.4f}, test_score:{:.4f}, auc:{:.4f}'.format(
            train_score, test_score, auc
        ))
        pyplt.xlabel('False Positive Rate')
        pyplt.ylabel('True Positive Rate')
        pyplt.plot(fpr, tpr)
        if show:
            pyplt.show()


if __name__ == '__main__':
    all_props_path = './../../resources/output/net/all_props.csv'
    run(all_props_path=all_props_path)
