import tensorflow as tf
import numpy as np
from data_input import import_train, import_valid, import_test
import matplotlib.pyplot as plt
from sklearn.utils import resample


EPOCHS = 1000

#Adapted from https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_tensorflow.py

TRAIN_NUM = 1500
VAL_NUM = 200

print('importing')
X_train, Y_train = import_train(TRAIN_NUM)
X_valid, Y_valid = import_valid(VAL_NUM)
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

Z_dim = 100
X_dim = 1024
y_dim = 46

xavier_init = tf.contrib.layers.xavier_initializer(uniform=False)

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    h1 = tf.layers.dense(
        inputs = inputs, units = 128,
        kernel_initializer=xavier_init,
        activation = tf.nn.relu, name = 'D_h1')
    logit = tf.layers.dense(
        kernel_initializer=xavier_init,
        inputs = h1, units = 1, name = 'D_logit')
    return logit

def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    h1 = tf.layers.dense(
        inputs = inputs, units = 128,
        kernel_initializer=xavier_init,
        activation = tf.nn.relu)
    log_prob = tf.layers.dense(
        kernel_initializer=xavier_init,
        inputs = h1, units = 1024)
    prob = tf.nn.sigmoid(log_prob)
    return prob

y = tf.placeholder('float', shape=(None, y_dim))

z = tf.placeholder('float', shape=(None, Z_dim))

x = tf.placeholder('float', shape=[None, X_dim])

with tf.variable_scope('Generator') as scope:
    generated_image = generator(z, y)

with tf.variable_scope('Discriminator') as scope:
    real_disc = discriminator(x, y)
    scope.reuse_variables()
    gen_disc = discriminator(generated_image, y)
    
disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_disc, labels=tf.ones_like(real_disc)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_disc, labels=tf.zeros_like(gen_disc)))

disc_loss = disc_loss_real+disc_loss_fake
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_disc, labels=tf.ones_like(gen_disc)))

var_list = [v for v in tf.trainable_variables() if v.name.startswith('Discriminator/')]
disc_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(disc_loss, var_list=var_list)

var_list = [v for v in tf.trainable_variables() if v.name.startswith('Generator/')]
gen_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(gen_loss, var_list=var_list)


N = 46*TRAIN_NUM
BATCH_SIZE = 300

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gen_image_vals = None
    for epoch in range(EPOCHS):
        print('Epoch: ' + str(epoch))
        shuffled_data, shuffled_labels = resample(X_train, Y_train, replace = False, n_samples = N)
        for iteration in range(int(N/BATCH_SIZE)):

            batch_data, batch_labels = shuffled_data[iteration*BATCH_SIZE:(iteration+1)*BATCH_SIZE], shuffled_labels[iteration*BATCH_SIZE:(iteration+1)*BATCH_SIZE]
            label_vals = np.zeros((BATCH_SIZE, 46))
            for i in range(BATCH_SIZE):
                label_vals[i,int(batch_labels[i])] = 1.0
            z_val = np.random.uniform(-1, 1, size = (BATCH_SIZE, Z_dim))
            
            _, disc_loss_val = sess.run([disc_optimizer, disc_loss],
                                        feed_dict={z: z_val, x: batch_data, y: label_vals})
            z_val = np.random.uniform(-1, 1, size = (BATCH_SIZE, Z_dim))
            _, gen_loss_val, gen_image_vals = sess.run([gen_optimizer, gen_loss, generated_image],
                                                       feed_dict={z: z_val, y: label_vals})

        print('Disc Loss: ' + str(disc_loss_val))
        print('Gen Loss: ' + str(gen_loss_val))

        if epoch == EPOCHS-1:
            
            images = [[0 for i in range(10)] for j in range(10)]
            
            label_vals = np.zeros((1, 46))
            label_vals[0,10] = 1.0
            z_val = np.random.uniform(-1, 1, size = (1, Z_dim))
            
            for i in range(5):
                for j in range(5):
                    z_val[0,0] = 2*i/5-1
                    z_val[0,1] = 2*j/5-1
                    gen_image_vals = sess.run(generated_image,
                                            feed_dict={z: z_val, y: label_vals})
                    images[i][j] = gen_image_vals[0,:]
            show_matrix(images)

            images = [[0 for i in range(10)] for j in range(10)]
            
            label_vals = np.zeros((1, 46))
            label_vals[0,3] = 1.0
            z_val = np.random.uniform(-1, 1, size = (1, Z_dim))
            
            for i in range(5):
                for j in range(5):
                    z_val[0,0] = 2*i/5-1
                    z_val[0,1] = 2*j/5-1
                    gen_image_vals = sess.run(generated_image,
                                            feed_dict={z: z_val, y: label_vals})
                    images[i][j] = gen_image_vals[0,:]
            show_matrix(images)








        
        
