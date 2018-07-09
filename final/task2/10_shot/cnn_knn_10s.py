import os
import argparse
import random
import pickle
import numpy as np
from scipy.misc import imread
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier


parser = argparse.ArgumentParser(description='Few-Shot CIFAR-100 Classification')
parser.add_argument('action', choices=['train_cnn', 'train_knn', 'test'])
parser.add_argument('-tr', '--train_data_dir', default='./task2-dataset/', type=str)
parser.add_argument('-te', '--test_data_dir', default='./test/')
parser.add_argument('-p', '--prediction_file', default='./prediction_10s.csv', type=str)
parser.add_argument('-s', '--save_model_dir', default='./saved_model/', type=str)
parser.add_argument('-lcnn', '--load_model_file', default='../model/cnn_model_10s.h5', type=str)
parser.add_argument('-lknn', '--knn_clf_file', default='../model/knn_clf_10s.pickle', type=str)
parser.add_argument('--cnn_num_classes', default=80, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--shot', default=10, type=int)
parser.add_argument('--neighbor', default=1, type=int)
parser.add_argument('-bs', '--best_seed', default=234, type=int)
args = parser.parse_args()


def load_base_data():
    x_train, y_train, x_valid, y_valid = [], [], [], []
    dir_list = sorted(os.listdir(args.train_data_dir + 'base'))
    for category in ['train', 'test']:
        for class_index, dir in enumerate(dir_list):
            imgs = sorted(os.listdir(args.train_data_dir + 'base/' + dir + '/' + category), key=lambda s: int(s[:-4]))
            for img in imgs:
                img_path = args.train_data_dir + 'base/' + dir + '/' + category + '/' + img
                img_data = imread(img_path) / 255.
                if category == 'train':
                    x_train.append(img_data)
                    y_train.append(class_index)
                else:
                    x_valid.append(img_data)
                    y_valid.append(class_index)
    x_train = np.array(x_train)
    y_train = to_categorical(np.array(y_train), args.cnn_num_classes)
    x_valid = np.array(x_valid)
    y_valid = to_categorical(np.array(y_valid), args.cnn_num_classes)

    return (x_train, y_train), (x_valid, y_valid)


def load_novel_data(seed):
    x_train, y_train, x_valid, y_valid = [], [], [], []
    dir_list = sorted(os.listdir(args.train_data_dir + 'novel'))
    for dir in dir_list:
        imgs = os.listdir(args.train_data_dir + 'novel/' + dir + '/train')
        random.seed(seed)
        random.shuffle(imgs)
        class_index = int(dir[6:])
        for count, img in enumerate(imgs):
            img_path = args.train_data_dir + 'novel/' + dir + '/train/' + img
            img_data = imread(img_path) / 255.
            if count < args.shot:
                x_train.append(img_data)
                y_train.append(class_index)
            else:
                x_valid.append(img_data)
                y_valid.append(class_index)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    return (x_train, y_train), (x_valid, y_valid) 


def load_test_data():
    x_test, index_test = [], []
    imgs = sorted(os.listdir(args.test_data_dir), key=lambda s: int(s[:-4]))
    for img in imgs:
        img_path = os.path.join(args.test_data_dir, img)
        img_data = imread(img_path) / 255.
        x_test.append(img_data)
        index_test.append(img[:-4])
    x_test = np.array(x_test)
    index_test = np.array(index_test)

    return (x_test, index_test)


def load_data(seed=0):
    if args.action == 'train_cnn':
        (x_base_train, y_base_train), (x_base_valid, y_base_valid) = load_base_data()

        ### random shuffle ###
        np.random.seed(0)
        indices = np.random.permutation(x_base_train.shape[0])
        x_base_train = x_base_train[indices]
        y_base_train = y_base_train[indices]

        return (x_base_train, y_base_train), (x_base_valid, y_base_valid)

    elif args.action == 'train_knn':
        (x_novel_train, y_novel_train), (x_novel_valid, y_novel_valid) = load_novel_data(seed)

        return (x_novel_train, y_novel_train), (x_novel_valid, y_novel_valid)

    elif args.action == 'test':
        (x_test, index_test) = load_test_data()

        return (x_test, index_test)


def CNN_model(extract=False):
    weight_decay = 0.0
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(args.cnn_num_classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)
    model.summary()
    if extract:
        extract_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        return extract_model

    return model


def main():
    if args.action == 'train_cnn':
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        # Shapes:
        #   x_base_train , y_base_train : (500*80, 32, 32, 3), (500*80, 80)
        #   x_base_valid , y_base_valid : (100*80, 32, 32, 3), (100*80, 80)
        (x_base_train, y_base_train), (x_base_valid, y_base_valid) = load_data()

        model = CNN_model()
        opt = Adam(1e-4)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        ckpt = ModelCheckpoint(os.path.join(args.save_model_dir, 'model_e{epoch:02d}_a{val_acc:.4f}.h5'),
                               monitor='val_acc',
                               mode='max',
                               save_best_only=True,
                               save_weights_only=True,
                               verbose=1)
        #es = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        callbacks_list = [ckpt]

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False) # randomly flip images

        datagen.fit(x_base_train)
        model.fit_generator(datagen.flow(x_base_train, y_base_train, batch_size=args.batch_size),
                            epochs=args.epochs,
                            verbose=1,
                            callbacks=callbacks_list,
                            validation_data=(x_base_valid, y_base_valid),
                            workers=4)


    elif args.action == 'train_knn':
        extract_model = CNN_model(extract=True)
        extract_model.load_weights(args.load_model_file, by_name=True)

        best_seed = 0
        best_score = 0.
        best_knn_clf = None
        for seed in random.sample(range(1000), 20):
            # Shapes:
            #   x_novel_train, y_novel_train: (      s*20, 32, 32, 3), (      s*20, 20)
            #   x_novel_valid, y_novel_valid: ((500-s)*20, 32, 32, 3), ((500-s)*20, 20)
            (x_novel_train, y_novel_train), (x_novel_valid, y_novel_valid) = load_data(seed)
            cnn_feature = extract_model.predict(x_novel_train)
            mean_feature = []
            mean_label = []
            if args.shot > 1:
                for i in range(20):
                    start_idx = i * args.shot
                    end_index = (i+1) * args.shot
                    mean_feature.append(np.mean(cnn_feature[start_idx:end_index], axis=0))
                    mean_label.append(y_novel_train[start_idx])
                mean_feature = np.array(mean_feature)
                mean_label = np.array(mean_label)
            else:
                mean_feature = cnn_feature
                mean_label = y_novel_train

            knn_clf = KNeighborsClassifier(n_neighbors=args.neighbor)
            knn_clf.fit(mean_feature, mean_label)

            score = knn_clf.score(extract_model.predict(x_novel_valid), y_novel_valid)
            print('seed: {}, score: {}'.format(seed, score))
            if score > best_score:
                best_seed = seed
                best_score = score
                best_knn_clf = knn_clf
        print('best seed: {}, best score: {}'.format(best_seed, best_score))
        pickle.dump(best_knn_clf, open(os.path.join(args.save_model_dir, 'knn_clf_10s.pickle'), 'wb'))
      

    elif args.action == 'test':
        best_seed = args.best_seed
        extract_model = CNN_model(extract=True)
        extract_model.load_weights(args.load_model_file, by_name=True)

        # Shapes:
        #   x_test, index_test: (2000, 32, 32, 3), (2000,)
        (x_test, index_test) = load_data(best_seed)

        knn_clf = pickle.load(open(args.knn_clf_file, 'rb'))
        pred = knn_clf.predict(extract_model.predict(x_test))

        with open(args.prediction_file, 'w') as fo:
            fo.write('image_id,predicted_label\n')
            for i in range(index_test.shape[0]):
                if pred[i] == 0:
                    fo.write('{},{}\n'.format(index_test[i], '00'))
                else:
                    fo.write('{},{}\n'.format(index_test[i], pred[i]))


if __name__ == '__main__':
    main()
