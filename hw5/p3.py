from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM
from keras.layers.wrappers import TimeDistributed
import os
import argparse
import numpy as np
from scipy.misc import imread


parser = argparse.ArgumentParser(description='Temporal Action Segmentation')
parser.add_argument('action', choices=['train', 'test'])
parser.add_argument('-tr', '--train_video', default='./HW5_data/FullLengthVideos/videos/train', type=str)
parser.add_argument('-v', '--valid_video', default='./HW5_data/FullLengthVideos/videos/valid', type=str)
parser.add_argument('-te', '--test_video', default='./HW5_data/FullLengthVideos/videos/valid', type=str)
parser.add_argument('-trl', '--train_label', default='./HW5_data/FullLengthVideos/labels/train', type=str)
parser.add_argument('-vl', '--valid_label', default='./HW5_data/FullLengthVideos/labels/valid', type=str)
parser.add_argument('-tel', '--test_label', default='./HW5_data/FullLengthVideos/labels/valid', type=str)
parser.add_argument('-o', '--output_dir', default='./output', type=str)

parser.add_argument('--save_model_dir', default='./save_model', type=str)
parser.add_argument('-l', '--load_model_file', default='./model/model_p3.h5', type=str)

parser.add_argument('--save_train_feature_dir', default='./feat_full_train', type=str)
parser.add_argument('--save_valid_feature_dir', default='./feat_full_valid', type=str)

parser.add_argument('--n_class', default=11, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)

args = parser.parse_args()


def build_classifier():
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(512, activation='tanh')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(128, activation='tanh')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(args.n_class, activation='softmax')))
    model.summary()
    return model


def main():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='max', input_shape=(240, 320, 3))
    base_model.trainable = False

    if args.action == 'train':

        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        x_train, y_train = [], []
        for v in sorted(os.listdir(args.train_video)):
            #one_video = []
            #for f in sorted(os.listdir(os.path.join(args.train_video, v))):
            #    frame = imread(os.path.join(os.path.join(args.train_video, v), f))
            #    one_video.append(frame)
            #one_video = np.array(one_video).astype(np.float32)
            #one_video = preprocess_input(one_video)
            #one_video = base_model.predict(one_video)
            #np.save(os.path.join(args.save_train_feature_dir, v) + '.npy', one_video)
            one_video = np.load(os.path.join(args.save_train_feature_dir, v+'.npy'))
            print(one_video.shape)
            x_train.append(one_video)
            y_train.append(np.genfromtxt(os.path.join(args.train_label, v+'.txt')))

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print()

        x_valid, y_valid = [], []
        for v in sorted(os.listdir(args.valid_video)):
            #one_video = []
            #for f in sorted(os.listdir(os.path.join(args.valid_video, v))):
            #    frame = imread(os.path.join(os.path.join(args.valid_video, v), f))
            #    one_video.append(frame)
            #one_video = np.array(one_video).astype(np.float32)
            #one_video = preprocess_input(one_video)
            #one_video = base_model.predict(one_video)
            #np.save(os.path.join(args.save_valid_feature_dir, v) + '.npy', one_video)
            one_video = np.load(os.path.join(args.save_valid_feature_dir, v+'.npy'))
            print(one_video.shape)
            x_valid.append(one_video)
            y_valid.append(np.genfromtxt(os.path.join(args.valid_label, v+'.txt')))

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

        ckpt = ModelCheckpoint(filepath=os.path.join(args.save_model_dir, 'model_p3_e{epoch:02d}_{val_acc:.4f}.h5'),
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

        if not os.path.exists('./p3_callback'):
            os.makedirs('./p3_callback')
        history.save('./p3_callback')


    elif args.action == 'test':
        x_test = []
        for v in sorted(os.listdir(args.test_video)):
            #one_video = []
            #for f in sorted(os.listdir(os.path.join(args.test_video, v))):
            #    frame = imread(os.path.join(os.path.join(args.test_video, v), f))
            #    one_video.append(frame)
            #one_video = np.array(one_video).astype(np.float32)
            #one_video = preprocess_input(one_video)
            #one_video = base_model.predict(one_video)
            one_video = np.load(os.path.join(args.save_valid_feature_dir, v) + '.npy')
            x_test.append(one_video)

        x_test = np.array(x_test)


if __name__ == '__main__':
    main()



