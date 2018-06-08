from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
import os
import sys
import argparse
import numpy as np
import random
import tensorflow as tf
from reader import getVideoList, readShortVideo


parser = argparse.ArgumentParser(description='CNN Feature Extraction')
parser.add_argument('action', choices=['train','test'])
parser.add_argument('-tr', '--train_video', default='./HW5_data/TrimmedVideos/video/train', type=str)
parser.add_argument('-v', '--valid_video', default='./HW5_data/TrimmedVideos/video/valid', type=str)
parser.add_argument('-te', '--test_video', default='./HW5_data/TrimmedVideos/video/valid', type=str)
parser.add_argument('-trl', '--train_label', default='./HW5_data/TrimmedVideos/label/gt_train.csv', type=str)
parser.add_argument('-vl', '--valid_label', default='./HW5_data/TrimmedVideos/label/gt_valid.csv', type=str)
parser.add_argument('-tel', '--test_label', default='./HW5_data/TrimmedVideos/label/gt_valid.csv', type=str)
parser.add_argument('-o', '--output_dir', default='./output/', type=str)
parser.add_argument('-on', '--output_name', default='p1_valid.txt', type=str)

parser.add_argument('--save_model_dir', default='./save_model/', type=str)
parser.add_argument('-l', '--load_model_file', default='./model/model_p1.h5', type=str)

parser.add_argument('--save_train_feature_dir', default='./feat_train', type=str)
parser.add_argument('--save_valid_feature_dir', default='./feat_valid', type=str)

parser.add_argument('--n_class', default=11, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=int)

args = parser.parse_args()


def build_classifier():
    fc1 = Dense(512, activation='relu')
    fc2 = Dense(128, activation='relu')
    classifier = Dense(args.n_class, activation='softmax')

    feature = Input(shape=(2048,))
    x = fc1(feature)
    x = Dropout(args.dropout_rate)(x)
    x = fc2(x)
    x = Dropout(args.dropout_rate)(x)
    category = classifier(x)

    model = Model(inputs=feature, outputs=category)
    #model.summary()

    return model


def main():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='max', input_shape=(240, 320, 3))
    base_model.trainable = False

    if args.action == 'train':
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        list_train = getVideoList(args.train_label)
        list_valid = getVideoList(args.valid_label)

        x_train, y_train = [], []
        for cate, name, label in zip(list_train['Video_category'], list_train['Video_name'], list_train['Action_labels']):
            frame = readShortVideo(args.train_video, cate, name)
            frame = preprocess_input(frame)
            feat = base_model.predict(frame)
            #np.save(os.path.join(args.save_train_feature_dir, name) + '.npy', feat)
            #feat = np.load(os.path.join(args.save_train_feature_dir, name) + '.npy')

            feat = np.mean(feat, axis=0)
            x_train.append(feat)
            y_train.append(np.eye(args.n_class, dtype=int)[int(label)])

        x_train = np.array(x_train)
        y_train = np.array(y_train)


        x_valid, y_valid = [], []
        for cate, name, label in zip(list_valid['Video_category'], list_valid['Video_name'], list_valid['Action_labels']):
            frame = readShortVideo(args.valid_video, cate, name)
            frame = preprocess_input(frame)
            feat = base_model.predict(frame)
            #np.save(os.path.join(args.save_valid_feature_dir, name) + '.npy', feat)
            #feat = np.load(os.path.join(args.save_valid_feature_dir, name) + '.npy')

            feat = np.mean(feat, axis=0)
            x_valid.append(feat)
            y_valid.append(np.eye(args.n_class, dtype=int)[int(label)])

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        classifier = build_classifier()
        classifier.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        class LossHistory(Callback):
            def __init__(self, train_data, valid_data):
                self.train_data = train_data
                self.valid_data = valid_data
                
            def on_train_begin(self, logs={}):
                self.losses = []
                self.val_losses = []
                self.acc = []
                self.val_acc = []

            def on_epoch_end(self, epoch, logs={}):
                x_valid = self.valid_data
                x_train = self.train_data
                self.losses.append(logs['loss'])
                self.val_losses.append(logs['val_loss'])
                self.acc.append(logs['acc'])
                self.val_acc.append(logs['val_acc'])
                
            def save(self, path):
                np.save(os.path.join(path, 'losses.npy'), self.losses)
                np.save(os.path.join(path, 'val_losses.npy'), self.val_losses)
                np.save(os.path.join(path, 'acc.npy'), self.acc)
                np.save(os.path.join(path, 'val_acc.npy'), self.val_acc)

        history = LossHistory(x_train, x_valid)

        ckpt = ModelCheckpoint(filepath=os.path.join(args.save_model_dir, 'model_p1_e{epoch:02d}_{val_acc:.4f}.h5'),
                               save_best_only=True,
                               save_weights_only=True,
                               verbose=1,
                               monitor='val_acc',
                               mode='max')

        classifier.fit(x_train,
                       y_train,
                       shuffle=True,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       validation_data=(x_valid, y_valid),
                       callbacks=[ckpt, history])

        if not os.path.exists('./p1_callback'):
            os.makedirs('./p1_callback')
        history.save('./p1_callback')

    elif args.action == 'test':
        list_test = getVideoList(args.test_label)

        x_test = []
        for cate, name, label in zip(list_test['Video_category'], list_test['Video_name'], list_test['Action_labels']):
            frame = readShortVideo(args.test_video, cate, name)
            frame = preprocess_input(frame)
            feat = base_model.predict(frame)
            #feat = np.load(os.path.join(args.save_valid_feature_dir, name) + '.npy')

            feat = np.mean(feat, axis=0)
            x_test.append(feat)

        x_test = np.array(x_test)

        classifier = build_classifier()
        classifier.load_weights(args.load_model_file)

        pred_prob = classifier.predict(x_test)
        pred = np.argmax(pred_prob, axis=-1)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(os.path.join(args.output_dir, args.output_name), 'w') as fo:
            for idx in range(pred.shape[0]):
                fo.write('{}\n'.format(pred[idx]))


if __name__ == '__main__':
    main()
