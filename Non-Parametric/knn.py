import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from data_input import import_train, import_valid, import_test
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import block_reduce
from sklearn.decomposition import PCA

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

def square(v):
    return v.reshape((32, 32))

def show(pixels):
    pixels = square(pixels)
    plt.imshow(pixels, cmap='gray')
    plt.show()

def get_points(u):
    u = square(u)
    points = []
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if u[i,j] >= 128:
                points.append((i,j))
    return np.asarray(points)

def euclidean(x1, x2):
    return np.linalg.norm(x1-x2)
 
def cosine(x1, x2):
    return 1-np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

print(X_valid.shape[0])
print(X_train.shape[0])


def pool(u, pool_size):
    return block_reduce(square(u),block_size = pool_size, func = np.median).flatten()

def pool_all(u, pool_size = (4,4)):
    return np.stack([pool(u[i,:], pool_size) for i in range(u.shape[0])], axis=0)


##print('Decision Tree')
##pool_sizes = [(1,1),(2,2),(4,4), (8,8)]
##for p in pool_sizes:
##    classifier = DecisionTreeClassifier(criterion = 'entropy', max_features = None)
##    classifier.fit(pool_all(get_X_train(), p), Y_train)
##    acc = classifier.score(pool_all(get_X_valid(), p), Y_valid)
##    print(p)
##    print(acc)
##
##print('Random Forests')
##pool_sizes = [(1,1),(2,2),(4,4), (8,8)]
##for p in pool_sizes:
##    classifier = RandomForestClassifier(criterion = 'entropy', max_features = None)
##    classifier.fit(pool_all(get_X_train(), p), Y_train)
##    acc = classifier.score(pool_all(get_X_valid(), p), Y_valid)
##    print(p)
##    print(acc)



print('KNN-PCA')
num_components = [10, 50, 100, 150]
ks = list(range(1,11))
for n_components in num_components:
    accs = []
    for k in range(1,11):
        pca = PCA(n_components = n_components)
        X_train_reduced = pca.fit_transform(X_train)
        X_valid_reduced = pca.transform(X_valid)
        classifier = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
        print('fitting')
        classifier.fit(X_train_reduced, Y_train)
        print('predicting')
        print(n_components)
        print(k)
        acc = classifier.score(X_valid_reduced, Y_valid)
        print(acc)
        accs.append(acc)
    plt.plot(ks, accs, label = str(n_components))
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()


##euclidean_accuracies  =[]
##cosine_accuracies = []
##ks = list(range(1,11))
##for k in range(1,11):
##    for n in [True, False]:
##        classifier = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
##        print('fitting')
##        classifier.fit(get_X_train(n), Y_train)
##        print('predicting')
##        print(k)
##        print(n)
##        acc = classifier.score(get_X_valid(n), Y_valid)
##        print(acc)
##        if n:
##            cosine_accuracies.append(acc)
##        else:
##            euclidean_accuracies.append(acc)
##plt.plot(ks, euclidean_accuracies, label = 'Euclidean Distance')
##plt.plot(ks, cosine_accuracies, label = 'Cosine Distance')
##plt.xlabel('k')
##plt.ylabel('Validation Accuracy')
##plt.legend()
##plt.show()

