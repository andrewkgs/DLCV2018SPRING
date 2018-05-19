from scipy.misc import imread, imsave
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def SetArgument():
    parser = argparse.ArgumentParser(description='Variational AutoEncoder')
    parser.add_argument('action', choices=['train', 'reconstruct', 'generate'])
    parser.add_argument('-d', '--data_path', default='./hw4_data/', type=str)
    parser.add_argument('-o', '--output_path', default='./output/', type=str)
    parser.add_argument('-s', '--save_model_dir', default='./save_model/', type=str)
    parser.add_argument('-l', '--load_model_file', default='./model/model_vae.ckpt', type=str)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--img_channels', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--latent_dim', default=1024, type=int)
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

def encoder_model(x_input, img_size, img_channels, latent_dim, isTrain):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(x_input, filters=64, kernel_size=4, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d(x, filters=512, kernel_size=4, strides=2, padding='same')
        x = tf.nn.leaky_relu(tf.layers.batch_normalization(x, training=isTrain))
        flat = tf.layers.flatten(x)
        mean = tf.layers.dense(flat, units=latent_dim)
        log_var = tf.layers.dense(flat, units=latent_dim)
        if isTrain:
            epsilon = tf.random_normal(tf.stack([tf.shape(flat)[0], latent_dim]))
        else:
            epsilon = tf.random_normal(tf.stack([tf.shape(flat)[0], latent_dim]), seed=0)
        z = mean + tf.multiply(epsilon, tf.exp(0.5 * log_var))
    return z, mean, log_var

def decoder_model(z_input, img_size, img_channels, latent_dim, isTrain):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(z_input, units=latent_dim)
        x = tf.reshape(x, [-1, 4, 4, latent_dim//16])
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, training=isTrain))
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = tf.nn.tanh(x)
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
    latent_dim = args.latent_dim

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # ==================== Load Data ==================== #

    if action == 'train':
        x_train, _ = LoadData(train_img_dir, train_label)
        #np.save('x_train.npy', x_train)
        #x_train = np.load('x_train.npy')
    else:
        x_test, y_test = LoadData(test_img_dir, test_label)

    print('\n=== finish loading data ===\n')

    graph = tf.Graph()
    with graph.as_default():
        x_input = tf.placeholder(shape=[None, img_size, img_size, img_channels], dtype=tf.float32, name='x_input')
        z_input = tf.placeholder(shape=[None, latent_dim], dtype=tf.float32, name='z_input')
        x_output = tf.placeholder(shape=[None, img_size, img_size, img_channels], dtype=tf.float32, name='x_output')

        # ==================== Train ==================== #

        if action == 'train':
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)

            z_sample, mean, log_var = encoder_model(x_input, img_size, img_channels, latent_dim, isTrain=True)
            x_dec = decoder_model(z_sample, img_size, img_channels, latent_dim, isTrain=True)

            recon_loss = tf.reduce_mean(tf.squared_difference(x_dec, x_output))
            KL_loss = (-0.5) * tf.reduce_sum(1.0 + log_var - tf.square(mean) - tf.exp(log_var))
            lamda = 2e-6
            loss = tf.reduce_sum(recon_loss + (lamda * KL_loss))

            optimizer = tf.train.AdamOptimizer()
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                train_op = optimizer.minimize(loss)

            saver = tf.train.Saver()

            with tf.Session(graph=graph) as sess:

                tf.summary.scalar('MSE loss', recon_loss)
                tf.summary.scalar('KLD loss', KL_loss)
                train_summary = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter('./logs_vae', graph=sess.graph)

                sess.run(tf.global_variables_initializer())

                max_iter = epochs * (len(x_train)//batch_size)
                print('max iteration: {}'.format(max_iter))

                for i in range(0, max_iter+1):
                    offset = i % (len(x_train)//batch_size)
                    batch = x_train[offset*batch_size : (offset+1)*batch_size, :, :, :]
                    _, recon, KL, total, summary = sess.run([train_op, recon_loss, KL_loss, loss, train_summary], feed_dict={x_input: batch, x_output: batch})
                    train_writer.add_summary(summary, i)

                    if i % 100 == 0:
                        print('iter: {:6}\trecon loss: {:.6}\tKLD loss: {:.6}\ttotal loss: {:.6}'.format(i, recon, KL, total))

                    if i % 2000 == 0:
                        saver.save(sess, os.path.join(args.save_model_dir, 'model_vae_{}.ckpt'.format(i)))

        # ==================== Recontruct ==================== #

        elif action == 'reconstruct':
            z_sample, mean, log_var = encoder_model(x_input, img_size, img_channels, latent_dim, isTrain=False)
            x_dec = decoder_model(z_sample, img_size, img_channels, latent_dim, isTrain=False)
            recon_loss = tf.squared_difference(x_dec, x_output)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, args.load_model_file)
            print('\n=== finish loading model ===\n')
            z_latent, img_recon, recon = sess.run([z_sample, x_dec, recon_loss], feed_dict={x_input: x_test, x_output: x_test})
            print('\nThe MSE of the entire testing set is {}'.format(recon.mean()))
            img_recon = np.array(img_recon)[0:10]
            z_latent = np.array(z_latent)

            origin_imgs = np.hstack(img for img in x_test[0:10]) * 127.5 + 127.5
            recon_imgs = np.hstack(img for img in img_recon) * 127.5 + 127.5
            whole_imgs = np.concatenate((origin_imgs, recon_imgs), axis=0)
            imsave(os.path.join(args.output_path, 'fig1_3.jpg'), whole_imgs)

            N = 200
            z_embed = TSNE(n_components=2, random_state=0).fit_transform(z_latent[:N])

            gender_0 = []
            gender_1 = []
            for i in range(N):
                if y_test[i, 7] == 0:
                    gender_0.append(i)
                else:
                    gender_1.append(i)

            plt.scatter(z_embed[gender_0, 0], z_embed[gender_0, 1], c='r', label='female')
            plt.scatter(z_embed[gender_1, 0], z_embed[gender_1, 1], c='b', label='male')
            plt.title('latent space map to 2D space')
            plt.legend(loc=0)
            plt.savefig(os.path.join(args.output_path, 'fig1_5.jpg'))
            #plt.show()
            
        # ==================== Generate ==================== #

        elif action == 'generate':
            np.random.seed(0)
            z_random = np.random.normal(size=(32, latent_dim))
            x_dec = decoder_model(z_input, img_size, img_channels, latent_dim, isTrain=False)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, args.load_model_file)
            print('\n=== finish loading model ===\n')
            img_decoded = sess.run([x_dec], feed_dict={z_input: z_random})
            img_decoded = np.squeeze(np.array(img_decoded), axis=0) * 127.5 + 127.5

            for i in range(4):
                if i == 0:
                    whole_imgs = np.hstack(img for img in img_decoded[8*i:8*(i+1)])
                else:
                    row = np.hstack(img for img in img_decoded[8*i:8*(i+1)])
                    whole_imgs = np.concatenate((whole_imgs, row), axis=0)
            imsave(os.path.join(args.output_path, 'fig1_4.jpg'), whole_imgs)
