from sklearn import svm
import csv
import numpy

w = 13.14
b = 5.21

x = 100 * numpy.random.rand(100)
y = ((100 * numpy.random.rand(100)) > 50).astype(numpy.int32)

x = x.reshape([-1, 1])

print('x.shape=', x.shape)
print('y.shape=', y.shape)

classifier = svm.SVC(C=1.0, kernel='linear')
classifier.fit(x, y)

print('score=', classifier.score(x, y))
