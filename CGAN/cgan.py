import tensorflow as tf
import numpy as np
from data_input import import_train, import_valid, import_test
import matplotlib.pyplot as plt
from sklearn.utils import resample

#Adapted from https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_tensorflow.py

print('importing')
X_train, Y_train = import_train(10)
X_valid, Y_valid = import_valid(10)
#Scale
X_train, X_valid = X_train/255, X_valid/255
#Flatten
Y_train, Y_valid = Y_train.ravel().astype(int), Y_valid.ravel().astype(int)

def square(v):
    return v.reshape((32, 32))

def show(pixels):
    pixels = square(pixels)
    print(np.max(pixels))
    print(np.min(pixels))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def show_matrix(images):
    fig, ax = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            pixels = square(images[i][j])
            ax[i,j].imshow(pixels, cmap='gray')
            ax[i,j].axis('off')
            if i+j == 0:
                print(np.min(pixels))
                print(np.max(pixels))
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.show()

mb_size = 64
Z_dim = 10
X_dim = 1024
y_dim = 46
h_dim = 128

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_logit

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

y = tf.placeholder('float', shape=(None, y_dim))

z = tf.placeholder('float', shape=(None, Z_dim))

x = tf.placeholder('float', shape=[None, X_dim])


generated_image = generator(z, y)
real_disc = discriminator(x, y)
gen_disc = discriminator(generated_image, y)

disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_disc, labels=tf.ones_like(real_disc)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_disc, labels=tf.zeros_like(gen_disc)))

disc_loss = disc_loss_real+disc_loss_fake
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_disc, labels=tf.ones_like(gen_disc)))

var_list = [v for v in tf.trainable_variables() if v.name.startswith('Discriminator/')]
disc_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(disc_loss, var_list=theta_D)

var_list = [v for v in tf.trainable_variables() if v.name.startswith('Generator/')]
gen_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(gen_loss, var_list=theta_G)

BATCH_SIZE = 100
ITERATIONS = 100000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gen_image_vals = None
    for iteration in range(ITERATIONS):
        print_now = False
        if iteration % 100 == 0:
            print_now = True

        if print_now:
            print('Iteration: ' + str(iteration))

        batch_data, batch_labels = resample(X_train, Y_train, n_samples = BATCH_SIZE)
        label_vals = np.zeros((BATCH_SIZE, 46))
        for i in range(BATCH_SIZE):
            label_vals[i,int(batch_labels[i])] = 1.0
        z_val = np.random.uniform(-1, 1, size = (BATCH_SIZE, Z_dim))
        
        _, disc_loss_val = sess.run([disc_optimizer, disc_loss],
                                    feed_dict={z: z_val, x: batch_data, y: label_vals})
        if print_now:
            print('Disc Loss: ' + str(disc_loss_val))
        
        _, gen_loss_val, gen_image_vals = sess.run([gen_optimizer, gen_loss, generated_image],
                                    feed_dict={z: z_val, y: label_vals})
        if print_now:
            print('Gen Loss: ' + str(gen_loss_val))
        
        if iteration % 10000 == 0:
            images = [[0 for i in range(10)] for j in range(10)]
            
            label_vals = np.zeros((BATCH_SIZE, 46))
            label_vals[i,10] = 1.0
            z_val = np.random.uniform(0, 1, size = (BATCH_SIZE, Z_dim))
            
            for i in range(5):
                for j in range(5):
                    z_val[0,0] = 2*i/5-1
                    z_val[0,1] = 2*j/5-1
                    gen_image_vals = sess.run(generated_image,
                                            feed_dict={z: z_val, y: label_vals})
                    images[i][j] = gen_image_vals[0,:]
            show_matrix(images)








        
        
