import numpy as np
from random import randint

def trainBatch(batchSize, maxLength, vectors):
    labels = []
    arr = np.zeros([batchSize, maxLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = vectors[num-1:num]
    return arr, labels

def testBatch(batchSize, maxLength, vectors):
    labels = []
    arr = np.zeros([batchSize, maxLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = vectors[num-1:num]
    return arr, labels