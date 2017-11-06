import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from data_input import import_train, import_valid, import_test
import matplotlib.pyplot as plt

print('importing')
X_train, Y_train = import_train()
X_valid, Y_valid = import_valid()
X_train_norm, X_valid_norm = normalize(X_train), normalize(X_valid)
#Flatten
Y_train, Y_valid = Y_train.ravel().astype(int), Y_valid.ravel().astype(int)

def get_X_train(norm = False):
    return X_train_norm if norm else X_train

def get_X_valid(norm = False):
    return X_valid_norm if norm else X_valid

def show(pixels):
    pixels = pixels.reshape((32,32))
    plt.imshow(pixels, cmap='gray')
    plt.show()

euclidean_accuracies  =[]
cosine_accuracies = []
ks = list(range(1,11))
for k in range(1,11):
    for n in [True, False]:
        classifier = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
        print('fitting')
        classifier.fit(get_X_train(n), Y_train)
        print('predicting')
        print(k)
        print(n)
        acc = classifier.score(get_X_valid(n), Y_valid)
        print(acc)
        if n:
            cosine_accuracies.append(acc)
        else:
            euclidean_accuracies.append(acc)
plt.plot(ks, euclidean_accuracies, label = 'Euclidean Distance')
plt.plot(ks, cosine_accuracies, label = 'Cosine Distance')
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()

