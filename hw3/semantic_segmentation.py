from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, Activation, Add
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
#from keras.utils.vis_utils import plot_model
from scipy.misc import imread, imsave
import os
import re
import argparse
import numpy as np


def SetArgument():
    parser = argparse.ArgumentParser(description='Semantic segmentation')
    parser.add_argument('action', choices=['train', 'test'])
    parser.add_argument('which_model', choices=['VGG16-FCN32s', 'VGG16-FCN16s'])
    parser.add_argument('--train_data_dir', default='./hw3-train-validation/train/', type=str)
    parser.add_argument('--valid_data_dir', default='./hw3-train-validation/validation/', type=str)
    parser.add_argument('--test_data_dir', default='./hw3-train-validation/validation/', type=str)
    parser.add_argument('--predict_dir', default='./prediction/', type=str)
    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--pretrained_model', default='./vgg16_weights_tf_dim_ordering_tf_kernels.h5', type=str)
    parser.add_argument('--model_dir', default='./new_model/', type=str)
    parser.add_argument('--model_file', default='./model/model.h5', type=str)
    return parser.parse_args()


def read_masks(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land
        masks[i, mask == 1] = 4  # (Blue: 001) Water
        masks[i, mask == 7] = 5  # (White: 111) Barren land
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
        masks[i, mask == 4] = 6  # (Red: 100) Unknown
    return masks


def LoadImage(path, load_mask, num_classes):
    img_files = sorted(os.listdir(path))
    img_sate, img_mask = [], []
    for img in img_files:
        if img.endswith('_sat.jpg'):
            img_sate.append(imread(os.path.join(path, img)))
    img_sate = np.array(img_sate)
    #img_sate[:,:,:,0] -= 103 #103.939
    #img_sate[:,:,:,1] -= 116 #116.779
    #img_sate[:,:,:,2] -= 123 #123.68

    if load_mask is True:
        img_mask = read_masks(path)
        img_mask = (np.arange(num_classes) == img_mask[:,:,:,None]).astype(np.int8)
        m_shape = img_mask.shape
        img_mask = np.reshape(img_mask, (m_shape[0], m_shape[1]*m_shape[2], m_shape[3]))
        return img_sate, img_mask

    mask_name = sorted([re.sub('sat.jpg', 'mask.png', sat) for sat in os.listdir(path) if sat.endswith('_sat.jpg')])
    return img_sate, mask_name


def VGG16FCN32s(num_classes):
    ### VGG16-FCN32s ###
    img_input = Input(shape=(512, 512, 3))
    weight_decay = 0
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

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fully_connected1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fully_connected2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, (1, 1), activation='linear', padding='valid', strides=(1, 1), name='fully_connected3', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

    x = Conv2DTranspose(num_classes, kernel_size=(64, 64), strides=(32, 32), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    x_shape = Model(img_input, x).output_shape
    output_h, output_w = x_shape[1], x_shape[2]
    x = Reshape((output_h*output_w, num_classes))(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    #model.summary()
    #plot_model(model, to_file='./VGG16-FCN32s.png', show_shapes=True, show_layer_names=True)

    return model


def VGG16FCN16s(num_classes):
    ### VGG16-FCN16s ###
    img_input = Input(shape=(512, 512, 3))
    weight_decay = 0
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

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    c_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(c_4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(4096, (7, 7), activation='relu', padding='same', kernel_regularizer=l2(weight_decay), name='fully_conv1')(x)
    x = Dropout(0.5)(x)
    c_7 = Conv2D(4096, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(weight_decay), name='fully_conv2')(x)
    c_7 = Dropout(0.5)(c_7)

    c_4 = Conv2D(num_classes, (1, 1), activation='linear', padding='valid', strides=(1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='fully_conv3')(c_4)
    c_7 = Conv2D(num_classes, (1, 1), activation='linear', padding='valid', strides=(1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='fully_conv4')(c_7)
    c_7 = Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(c_7)

    o = Add()([c_4, c_7])

    o = Conv2DTranspose(num_classes, kernel_size=(32, 32), strides=(16, 16), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(o)

    o_shape = Model(img_input, o).output_shape
    output_h, output_w = o_shape[1], o_shape[2]
    o = Reshape((output_h*output_w, num_classes))(o)
    o = Activation('softmax')(o)

    model = Model(img_input, o)
    #model.summary()
    #plot_model(model, to_file='./VGG16-FCN16s.png', show_shapes=True, show_layer_names=True)

    return model


def train(x_train, y_train, x_valid, y_valid, batch_size, epochs, num_classes, model_dir, which_model, pretrained_model):
    if which_model == 'VGG16-FCN32s':
        model = VGG16FCN32s(num_classes)
    elif which_model == 'VGG16-FCN16s':
        model = VGG16FCN16s(num_classes)
    
    model.load_weights(pretrained_model, by_name=True)

    opt = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'model_e{epoch:02d}_a{val_acc:.4f}.h5'), monitor='val_acc', save_weights_only=True)
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks_list, validation_data=(x_valid, y_valid))


def test(x_test, mask_list, predict_dir, num_classes, model_file, which_model):
    if which_model == 'VGG16-FCN32s':
        model = VGG16FCN32s(num_classes)
    elif which_model == 'VGG16-FCN16s':
        model = VGG16FCN16s(num_classes)
    model.load_weights(model_file, by_name=True)

    pred = model.predict(x_test, batch_size=4)
    pred = np.argmax(np.squeeze(pred), axis=-1).astype(np.uint8)
    pred = np.reshape(pred, (pred.shape[0], 512, 512))

    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)

    mask = np.empty((512, 512, 3)).astype(np.uint8)
    for i in range(len(mask_list)):
        mask[pred[i, :, :] == 0] = [  0, 255, 255]
        mask[pred[i, :, :] == 1] = [255, 255,   0]
        mask[pred[i, :, :] == 2] = [255,   0, 255]
        mask[pred[i, :, :] == 3] = [  0, 255,   0]
        mask[pred[i, :, :] == 4] = [  0,   0, 255]
        mask[pred[i, :, :] == 5] = [255, 255, 255]
        mask[pred[i, :, :] == 6] = [  0,   0,   0]
        imsave(os.path.join(predict_dir, mask_list[i]), mask)


if __name__ == '__main__':

    args = SetArgument()

    if args.action == 'train':
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        npy_list = ['./img_npy/x_train.npy', './img_npy/y_train.npy', './img_npy/x_valid.npy', './img_npy/y_valid.npy']
        npy_exist = 0
        for file in npy_list:
            if os.path.exists(file):
                npy_exist += 1
            else:
                break
        if npy_exist == 4:
            x_train, y_train = np.load('./img_npy/x_train.npy'), np.load('./img_npy/y_train.npy')
            x_valid, y_valid = np.load('./img_npy/x_valid.npy'), np.load('./img_npy/y_valid.npy')
        else:
            x_train, y_train = LoadImage(args.train_data_dir, load_mask=True, num_classes=args.num_classes)
            x_valid, y_valid = LoadImage(args.valid_data_dir, load_mask=True, num_classes=args.num_classes)
            if not  os.path.exists('./img_npy/'):
                os.makedirs('./img_npy/')
            #np.save('./img_npy/x_train.npy', x_train)
            #np.save('./img_npy/y_train.npy', y_train)
            #np.save('./img_npy/x_valid.npy', x_valid)
            #np.save('./img_npy/y_valid.npy', y_valid)
        train(x_train, y_train, x_valid, y_valid, args.batch_size, args.epochs, args.num_classes, args.model_dir, args.which_model, args.pretrained_model)

    elif args.action == 'test':
        if not os.path.exists(args.predict_dir):
            os.makedirs(args.predict_dir)
        x_test, mask_list = LoadImage(args.test_data_dir, load_mask=False, num_classes=args.num_classes)
        test(x_test, mask_list, args.predict_dir, args.num_classes, args.model_file, args.which_model)

