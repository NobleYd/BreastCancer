import numpy
import os
import os.path

arr1 = [1, 2, 3, 4, 5]
print(arr1)

arr2 = [e + 1 for e in arr1]
print(arr2)

print(os.path.join('a', 'b', 'c'))

print('a.b'.split('.')[0])

list1 = [1, 2, 3]
list2 = [4, 5, 6]

list3 = [list1, list2]
list4 = zip(*list3)
print(list(list4))
