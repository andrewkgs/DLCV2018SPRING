from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
import os
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
parser.add_argument('-on', '--output_name', default='p2_result.txt', type=str)

parser.add_argument('--save_model_dir', default='./save_model/', type=str)
parser.add_argument('-l', '--load_model_file', default='./model/model_p2.h5', type=str)

parser.add_argument('--save_train_feature_dir', default='./feat_train', type=str)
parser.add_argument('--save_valid_feature_dir', default='./feat_valid', type=str)

parser.add_argument('--n_class', default=11, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seq_max_len', default=50, type=int)
parser.add_argument('--dropout_rate', default=0.4, type=int)

args = parser.parse_args()


def build_classifier():
    re1 = LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, activation='tanh')
    re2 = LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, activation='tanh')
    re3 = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, activation='tanh', name='rnn_output')
    fc1 = Dense(512, activation='relu')
    fc2 = Dense(128, activation='relu')
    classifier = Dense(11, activation='softmax')

    feature = Input(shape=(args.seq_max_len, 2048))
    x = re1(feature)
    x = re2(x)
    x = re3(x)
    x = fc1(x)
    x = Dropout(args.dropout_rate)(x)
    x = fc2(x)
    x = Dropout(args.dropout_rate)(x)
    category = classifier(x)

    model = Model(inputs=feature, outputs=category)
    rnn_model = Model(inputs=model.input, outputs=model.get_layer('rnn_output').output)
    #model.summary()

    return model, rnn_model


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

            x_train.append(feat)
            y_train.append(np.eye(args.n_class, dtype=int)[int(label)])

        x_train = np.array(x_train)
        x_train = pad_sequences(x_train, maxlen=args.seq_max_len)
        y_train = np.array(y_train)


        x_valid, y_valid = [], []
        for cate, name, label in zip(list_valid['Video_category'], list_valid['Video_name'], list_valid['Action_labels']):
            frame = readShortVideo(args.train_video, cate, name)
            frame = preprocess_input(frame)
            feat = base_model.predict(frame)
            #np.save(os.path.join(args.save_valid_feature_dir, name) + '.npy', feat)
            #feat = np.load(os.path.join(args.save_valid_feature_dir, name) + '.npy')

            x_valid.append(feat)
            y_valid.append(np.eye(args.n_class, dtype=int)[int(label)])

        x_valid = np.array(x_valid)
        x_valid = pad_sequences(x_valid, maxlen=args.seq_max_len)
        y_valid = np.array(y_valid)

        classifier, _ = build_classifier()
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

        ckpt = ModelCheckpoint(filepath=os.path.join(args.save_model_dir, 'model_p2_e{epoch:02d}_{val_acc:.4f}.h5'),
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

        if not os.path.exists('./p2_callback'):
            os.makedirs('./p2_callback')
        history.save('./p2_callback')

    elif args.action == 'test':
        list_test = getVideoList(args.test_label)

        x_test = []
        #x_test_mean = []
        for cate, name, label in zip(list_test['Video_category'], list_test['Video_name'], list_test['Action_labels']):
            frame = readShortVideo(args.test_video, cate, name)
            frame = preprocess_input(frame)
            feat = base_model.predict(frame)
            #feat = np.load(os.path.join(args.save_valid_feature_dir, name) + '.npy')
            x_test.append(feat)

            #feat_mean = np.mean(feat, axis=0)
            #x_test_mean.append(feat_mean)

        x_test = np.array(x_test)
        x_test = pad_sequences(x_test, maxlen=args.seq_max_len)
        #cnn_feature = np.array(x_test_mean)
        #np.save('./cnn_feature.npy', cnn_feature)

        classifier, rnn = build_classifier()
        classifier.load_weights(args.load_model_file)

        pred_prob = classifier.predict(x_test)
        #rnn_feature = rnn.predict(x_test)
        #np.save('./rnn_feature.npy', rnn_feature)
        pred = np.argmax(pred_prob, axis=-1)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(os.path.join(args.output_dir, args.output_name), 'w') as fo:
            for idx in range(pred.shape[0]):
                fo.write('{}\n'.format(pred[idx]))


if __name__ == '__main__':
    main()
