import numpy as np
from PIL import Image
import os

IMG_SIZE = 1024 # num pixels in image
IMG_DIM = (32, 32) 

# set num to the number of examples you want per image. Default is all
def import_train(num = None):
  chars = {}
  n = 0
  dirs = [x[0] for x in os.walk("./Train")][1:]
  for d in dirs:
    if d[8] == 'c':
      c = d.split('_', 1)[1]
    else:
      c = d.split('/')[2]
    imgs = os.listdir(d)
    if num is None:
      chars[c] = np.zeros((len(imgs), IMG_SIZE))
    else:
      chars[c] = np.zeros((num, IMG_SIZE))
      imgs = imgs[:num]
    n += len(imgs)
    for i in range(len(imgs)):
      chars[c][i,] = np.array(Image.open(d + '/' + imgs[i]).convert('L')).ravel()

  X = np.zeros((n, IMG_SIZE))
  Y = []
  count = 0
  for char in chars:
    X[count:count+chars[char].shape[0], ] = chars[char]
    Y += [char]*chars[char].shape[0]
    count += chars[char].shape[0]
  Y = np.array(Y, dtype = np.dtype('a10')).reshape((len(Y), 1))

  return (X, Y)

