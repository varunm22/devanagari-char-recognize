import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from data_input import *

# mode is 'ovo' (one vs one) or 'ovr' (one vs rest)

def main(mode='ovo'):
  nums = [10, 20, 50, 100, 200, 500, 1000, 1500]
  Cs = [0.01, 1, 100]
  acc = np.zeros((len(nums), len(Cs)))
  for i in range(len(nums)):
    num = nums[i]
    trainX, trainY = import_train(num)
    validX, validY = import_valid()
    for j in range(len(Cs)):
      C = Cs[j]
      if mode=='ovo':
        svm = SVC(kernel='linear', C=C)
      else:
        svm = LinearSVC(C=C)
    
      model = svm.fit(trainX, trainY)
      pred = model.predict(validX)
      accuracy = model.score(validX, validY)
      cm = confusion_matrix(validY, pred)
    
      print num, C, accuracy
      acc[i,j] = accuracy
    print acc

main(mode='ovr')
