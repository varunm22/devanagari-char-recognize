import numpy as np
from PIL import Image
import os

IMG_SIZE = 1024 # num pixels in image
IMG_DIM = (32, 32) 

# training data has 1700 examples per char. We're setting 200 aside for validation
# set param num to the number of examples you want per image, default/max is 1500
def import_train(num = 1500):
  return import_data(0, min(num, 1500), "./Data/Train")

# same meaning for param num, default/max is 200
def import_valid(num = 200):
  return import_data(1500, min(num, 200), "./Data/Train")

# same meaning for param num, default/max is 300
def import_test(num = 300):
  return import_data(0, min(num, 300), "./Data/Test")

def import_data(start, num, path):
  chars = {}
  dirs = [x[0] for x in os.walk(path)][1:]
  for d in dirs:
    if d[13] == 'c':
      c = int(d.split('_')[1]) + 10
    else:
      c = int(d.split('_')[1])
    imgs = os.listdir(d)[start:start+num]
    chars[c] = np.zeros((num, IMG_SIZE))
    for i in range(len(imgs)):
      chars[c][i,] = np.array(Image.open(d + '/' + imgs[i]).convert('L')).ravel()

  X = np.zeros((num*len(dirs), IMG_SIZE))
  Y = np.zeros(num*len(dirs))
  count = 0
  for char in chars:
    X[count:count+chars[char].shape[0], ] = chars[char]
    Y[count:count+chars[char].shape[0]] = char 
    count += chars[char].shape[0]

  return (X, Y)
