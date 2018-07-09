import os
import argparse
import numpy as np
from scipy.misc import imread
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


parser = argparse.ArgumentParser(description='Fashion MNIST Classification')
parser.add_argument('action', choices=['train', 'test'])
parser.add_argument('-tr', '--train_data_dir', default='./Fashion_MNIST_student/train/', type=str)
parser.add_argument('-te', '--test_data_dir', default='./Fashion_MNIST_student/test/', type=str)
parser.add_argument('-p', '--prediction_file', default='./prediction.csv', type=str)
parser.add_argument('-s', '--save_model_dir', default='./saved_model/', type=str)
parser.add_argument('-l', '--load_model_file', default='./model/model_task_1.h5', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--valid_ratio', default=0.1, type=float)

args = parser.parse_args()


def load_data():
    if args.action == 'train':
        x_train, y_train = [], []
        dir_list = sorted(os.listdir(args.train_data_dir))
        for i, dir in enumerate(dir_list):
            imgs = sorted(os.listdir(os.path.join(args.train_data_dir, dir)), key=lambda s: int(s[:-4]))
            for img in imgs:
                img_path = os.path.join(args.train_data_dir, os.path.join(dir, img))
                img_data = imread(img_path)
                mean = np.mean(img_data)
                std = np.std(img_data)
                x_train.append((img_data-mean)/std)
                y_train.append(i)
        x_train = np.expand_dims(np.array(x_train), axis=-1) # shape = (2000, 28, 28, 1)
        y_train = np.array(y_train) # shape = (2000,)
        y_train = to_categorical(y_train, args.num_classes) # shape = (2000, 10)

        ### random shuffle ###
        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]

        split_num = int(x_train.shape[0]*(1.0-args.valid_ratio))

        return x_train[:split_num], y_train[:split_num], x_train[split_num:], y_train[split_num:]


    elif args.action == 'test':
        x_test, index_test = [], []
        imgs = sorted(os.listdir(args.test_data_dir), key=lambda s: int(s[:-4]))
        for img in imgs:
            img_path = os.path.join(args.test_data_dir, img)
            img_data = imread(img_path)
            mean = np.mean(img_data)
            std = np.std(img_data)
            x_test.append((img_data-mean)/std)
            index_test.append(img[:-4])
        x_test = np.expand_dims(np.array(x_test), axis=-1) # shape = (10000, 28, 28, 1)
        index_test = np.array(index_test)

        return x_test, index_test


def CNN_model():
    weight_decay = 0.0
    img_input = Input(shape=(28, 28, 1))
    x = Conv2D(64, (2, 2), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (2, 2), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (2, 2), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (2, 2), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (2, 2), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (2, 2), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(2048, name='fc1', activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, name='fc2', activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(args.num_classes, activation='softmax')(x)

    model = Model(img_input, x)
    model.summary()

    return model


def main():
    if args.action == 'train':
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        x_train, y_train, x_valid, y_valid = load_data()

        model = CNN_model()
        opt = Adam(lr=1e-4)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        ckpt = ModelCheckpoint(os.path.join(args.save_model_dir, 'model_e{epoch:02d}_a{val_acc:.4f}.h5'),
                               monitor='val_acc',
                               mode='max',
                               save_best_only=False,
                               save_weights_only=True,
                               verbose=1)
        es = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        callbacks_list = [ckpt, es]

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.03,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.03,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False) # randomly flip images

        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                            epochs=args.epochs,
                            verbose=1,
                            callbacks=callbacks_list,
                            validation_data=(x_valid, y_valid),
                            workers=4)

    elif args.action == 'test':
        x_test, index_test = load_data()

        model = CNN_model()
        model.load_weights(args.load_model_file)
        prediction = model.predict(x_test)
        prediction = np.argmax(prediction, axis=-1)

        with open(args.prediction_file, 'w') as fo:
            fo.write('image_id,predicted_label\n')
            for i in range(prediction.shape[0]):
                fo.write('{},{}\n'.format(index_test[i], prediction[i]))


if __name__ == '__main__':
    main()
