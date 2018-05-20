from scipy.misc import imread, imsave
import os
import re
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def SetArgument():
    parser = argparse.ArgumentParser(description='Generative Adversarial Network')
    parser.add_argument('action', choices=['train', 'generate'])
    parser.add_argument('-d', '--data_path', default='./hw4_data/', type=str)
    parser.add_argument('-o', '--output_path', default='./output/', type=str)
    parser.add_argument('-s', '--save_model_dir', default='./save_model/', type=str)
    parser.add_argument('-l', '--load_model_file', default='./model/model_dcgan.ckpt', type=str)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--img_channels', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--noise_dim', default=100, type=int)
    return parser.parse_args()


def LoadData(img_path, label_path):
    img_files = sorted(os.listdir(img_path))
    images = []
    for img in img_files:
        images.append(imread(os.path.join(img_path, img)))
    images = (np.array(images).astype('float32') - 127.5) / 127.5
    labels = np.genfromtxt(label_path, delimiter=',', dtype=np.int8)[1:, 1:]
    return images, labels

# ==================== Model ==================== #

def generator(z_input, isTrain, reuse=False):
    regu = tf.contrib.layers.l2_regularizer(scale=0.1)
    with tf.variable_scope('G', reuse=reuse):
        x = tf.layers.dense(z_input, units=4096)
        x = tf.reshape(x, shape=[-1, 2, 2, 1024])
        x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.tanh(x)
        return x

def discriminator(x_input, isTrain, reuse=False):
    regu = tf.contrib.layers.l2_regularizer(scale=0.1)
    with tf.variable_scope('D', reuse=reuse):
        x = tf.layers.conv2d(x_input, filters=64, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)     
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d(x, filters=256, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d(x, filters=512, kernel_size=5, strides=2, padding='same', kernel_regularizer=regu)
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=1, activation=None)
        x = tf.nn.leaky_relu(x)
        return x


if __name__ == '__main__':

    args = SetArgument()

    train_img_dir = os.path.join(args.data_path, 'train/')
    train_label = os.path.join(args.data_path, 'train.csv')
    test_img_dir = os.path.join(args.data_path, 'test/')
    test_label = os.path.join(args.data_path, 'test.csv')

    action = args.action
    img_size = args.img_size
    img_channels = args.img_channels
    epochs = args.epochs
    batch_size = args.batch_size
    noise_dim = args.noise_dim
    
    if action == 'train':
        x_train, _ = LoadData(train_img_dir, train_label)
        #np.save('x_train.npy', x_train)
        #x_train = np.load('x_train.npy')

    print('\n=== finish loading data ===\n')

    with tf.variable_scope('vars'):
        real_image = tf.placeholder(shape=[None, img_size, img_size, img_channels], dtype=tf.float32, name='real_image')
        noise_input = tf.placeholder(shape=[None, noise_dim], dtype=tf.float32, name='noise_input')
        isTrain = tf.placeholder(shape=[], dtype=tf.bool, name='isTrain')

        G_sample = generator(noise_input, isTrain=isTrain)

        D_real = discriminator(real_image, isTrain=isTrain)
        D_fake = discriminator(G_sample, isTrain=isTrain, reuse=True)

        G_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='G_target')
        D_r_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='D_r_target')
        D_f_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='D_f_target')

        G_loss = tf.losses.sigmoid_cross_entropy(G_target, logits=D_fake)
        D_r_loss = tf.losses.sigmoid_cross_entropy(D_r_target, logits=D_real)
        D_f_loss = tf.losses.sigmoid_cross_entropy(D_f_target, logits=D_fake)
        D_loss = D_r_loss + D_f_loss

        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vars/G/')
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vars/D/')

        G_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
        D_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)

        G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='vars/G/')
        D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='vars/D/')

        with tf.control_dependencies(G_update_ops):
            G_train = G_optimizer.minimize(G_loss, var_list=G_vars)
        with tf.control_dependencies(D_update_ops):
            D_train = D_optimizer.minimize(D_loss, var_list=D_vars)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # ==================== Train ==================== #

        if args.action == 'train':

            if not os.path.exists('./imgs'):
                os.makedirs('./imgs')
            
            D_r_loss_sum = tf.summary.scalar('D r loss', D_r_loss)
            D_f_loss_sum = tf.summary.scalar('D f loss', D_f_loss)
            D_loss_sum = tf.summary.scalar('D loss', D_loss)
            G_loss_sum = tf.summary.scalar('G loss', G_loss)
            train_sum = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./logs_dcgan', graph=sess.graph)
            
            max_iter = epochs * (len(x_train)//batch_size)
            print('max iteration: {}'.format(max_iter))

            for i in range(0, max_iter+1):
                offset = i % (len(x_train)//batch_size)
                batch_x = x_train[offset*batch_size : (offset+1)*batch_size, :, :, :]

                z_random = np.random.normal(size=[batch_size, noise_dim])

                D_r_batch_y = np.ones([batch_size, 1])
                D_f_batch_y = np.zeros([batch_size, 1])
                G_batch_y = np.ones([batch_size, 1])

                feed_dict = {real_image: batch_x,
                             noise_input: z_random,
                             D_r_target: D_r_batch_y,
                             D_f_target: D_f_batch_y,
                             G_target: G_batch_y,
                             isTrain: True}

                _, d_loss, summary = sess.run([D_train, D_loss, train_sum], feed_dict=feed_dict)
                train_writer.add_summary(summary, i)
                _, g_loss, summary = sess.run([G_train, G_loss, train_sum], feed_dict=feed_dict)
                train_writer.add_summary(summary, i)

                if i % 200 == 0:
                    print('iter: {:6}\tG loss: {:.7}\tD loss: {:.7}'.format(i, g_loss, d_loss))

                    z_random = np.random.normal(size=[32, noise_dim])

                    image_generate = sess.run(G_sample, feed_dict={noise_input: z_random, isTrain: False})
                    image_generate = image_generate * 127.5 + 127.5
                    for j in range(4):
                        if j == 0:
                            whole_imgs = np.hstack(img for img in image_generate[8*j:8*(j+1)])
                        else:
                            row = np.hstack(img for img in image_generate[8*j:8*(j+1)])
                            whole_imgs = np.concatenate((whole_imgs, row), axis=0)
                    imsave('./imgs/{}.png'.format(i), whole_imgs)

                if i % 2000 == 0:
                    saver.save(sess, os.path.join(args.save_model_dir, 'model_dcgan_{}.ckpt'.format(i)))

        # ==================== Generate ==================== #

        elif args.action == 'generate':

            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)

            np.random.seed(1)
            z_random = np.random.normal(size=[32, noise_dim])

            saver.restore(sess, args.load_model_file)
            print('\n=== finish loading model ===\n')

            image_generate = sess.run(G_sample, feed_dict={noise_input: z_random, isTrain: False})
            image_generate = image_generate * 127.5 + 127.5

            for i in range(4):
                if i == 0:
                    whole_imgs = np.hstack(img for img in image_generate[8*i:8*(i+1)])
                else:
                    row = np.hstack(img for img in image_generate[8*i:8*(i+1)])
                    whole_imgs = np.concatenate((whole_imgs, row), axis=0)
            imsave(os.path.join(args.output_path, 'fig2_3.jpg'), whole_imgs)


