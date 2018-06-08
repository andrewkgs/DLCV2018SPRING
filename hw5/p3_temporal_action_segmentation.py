from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM
from keras.utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import Callback, ModelCheckpoint
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
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--seq_max_len', default=250, type=int)

args = parser.parse_args()


def build_classifier():
    model = Sequential()
    model.add(LSTM(256, input_shape=(args.seq_max_len, 2048), return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(512, activation='tanh')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(128, activation='tanh')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(args.n_class, activation='softmax')))
    #model.summary()
    return model


def main():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='max', input_shape=(240, 320, 3))
    base_model.trainable = False

    if args.action == 'train':

        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)

        _x_train, _y_train = [], []
        for v in sorted(os.listdir(args.train_video)):
            one_video = []
            for f in sorted(os.listdir(os.path.join(args.train_video, v))):
                frame = imread(os.path.join(os.path.join(args.train_video, v), f))
                one_video.append(frame)
            one_video = np.array(one_video).astype(np.float32)
            one_video = preprocess_input(one_video)
            one_video = base_model.predict(one_video)
            #np.save(os.path.join(args.save_train_feature_dir, v) + '.npy', one_video)
            #one_video = np.load(os.path.join(args.save_train_feature_dir, v+'.npy'))
            _x_train.append(one_video)
            _y_train.append(np.genfromtxt(os.path.join(args.train_label, v+'.txt')))

        _x_train = np.array(_x_train)
        _y_train = np.array(_y_train)

        _x_valid, _y_valid = [], []
        for v in sorted(os.listdir(args.valid_video)):
            one_video = []
            for f in sorted(os.listdir(os.path.join(args.valid_video, v))):
                frame = imread(os.path.join(os.path.join(args.valid_video, v), f))
                one_video.append(frame)
            one_video = np.array(one_video).astype(np.float32)
            one_video = preprocess_input(one_video)
            one_video = base_model.predict(one_video)
            #np.save(os.path.join(args.save_valid_feature_dir, v) + '.npy', one_video)
            #one_video = np.load(os.path.join(args.save_valid_feature_dir, v+'.npy'))
            _x_valid.append(one_video)
            _y_valid.append(np.genfromtxt(os.path.join(args.valid_label, v+'.txt')))

        _x_valid = np.array(_x_valid)
        _y_valid = np.array(_y_valid)

        x_train, y_train = [], []
        for i in range(_x_train.shape[0]):
            N = int(np.ceil(_x_train[i].shape[0]/args.seq_max_len))
            for j in range(N):
                if not j == N-1:
                    start_idx = j*args.seq_max_len
                    end_idx = start_idx + args.seq_max_len
                    x_train.append(_x_train[i][start_idx : end_idx])
                    y_train.append(_y_train[i][start_idx : end_idx])
                else:
                    end_idx = _x_train[i].shape[0]
                    start_idx = end_idx - args.seq_max_len
                    x_train.append(_x_train[i][start_idx : end_idx])
                    y_train.append(_y_train[i][start_idx : end_idx])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_train = to_categorical(y_train, args.n_class)

        x_valid, y_valid = [], []
        for i in range(_x_valid.shape[0]):
            N = int(np.ceil(_x_valid[i].shape[0]/args.seq_max_len))
            for j in range(N):
                if not j == N-1:
                    start_idx = j*args.seq_max_len
                    end_idx = start_idx + args.seq_max_len
                    x_valid.append(_x_valid[i][start_idx : end_idx])
                    y_valid.append(_y_valid[i][start_idx : end_idx])
                else:
                    end_idx = _x_valid[i].shape[0]
                    start_idx = end_idx - args.seq_max_len
                    x_valid.append(_x_valid[i][start_idx : end_idx])
                    y_valid.append(_y_valid[i][start_idx : end_idx])
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)
        y_valid = to_categorical(y_valid, args.n_class)


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
        video_name, video_length, video_num = [], [], []
        _x_test = []
        for v in sorted(os.listdir(args.test_video)):
            one_video = []
            for f in sorted(os.listdir(os.path.join(args.test_video, v))):
                frame = imread(os.path.join(os.path.join(args.test_video, v), f))
                one_video.append(frame)
            one_video = np.array(one_video).astype(np.float32)
            one_video = preprocess_input(one_video)
            one_video = base_model.predict(one_video)
            #one_video = np.load(os.path.join(args.save_valid_feature_dir, v+'.npy'))
            _x_test.append(one_video)

            video_name.append(v)
            video_length.append(one_video.shape[0])

        _x_test = np.array(_x_test)

        x_test = []
        last_video_start_index = []
        for i in range(_x_test.shape[0]):
            N = int(np.ceil(_x_test[i].shape[0]/args.seq_max_len))
            video_num.append(N)
            for j in range(N):
                if not j == N-1:
                    start_idx = j*args.seq_max_len
                    end_idx = start_idx + args.seq_max_len
                    x_test.append(_x_test[i][start_idx : end_idx])
                else:
                    end_idx = _x_test[i].shape[0]
                    start_idx = end_idx - args.seq_max_len
                    last_video_start_index.append(start_idx)
                    x_test.append(_x_test[i][start_idx : end_idx])
        x_test = np.array(x_test)

        classifier = build_classifier()
        classifier.load_weights(args.load_model_file)

        pred_prob = classifier.predict(x_test)
        pred = np.argmax(pred_prob, axis=-1)

        video_end_index = []
        for i in range(len(video_num)):
            video_end_index.append(sum(video_num[0:i+1]))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for i, vn in enumerate(video_name):
            prediction = []
            if i == 0:
                prediction.append(pred[0: video_end_index[i]])

            else:
                prediction.append(pred[video_end_index[i-1]: video_end_index[i]])

            with open(os.path.join(args.output_dir, vn+'.txt'), 'w') as fo:
                for pvs in prediction:
                    for j, pv in enumerate(pvs):
                        if not j == video_num[i] - 1:
                            for pf in pv:
                                fo.write('{}\n'.format(pf))
                        else:
                            for pf in pv[args.seq_max_len-(last_video_start_index[i]%args.seq_max_len):]:
                                fo.write('{}\n'.format(pf))


if __name__ == '__main__':
    main()

