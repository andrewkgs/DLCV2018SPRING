from scipy.io import loadmat
from scipy.signal import convolve2d
from skimage import io
from skimage.color import label2rgb, rgb2lab, rgb2gray
from sklearn.cluster import KMeans
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='DLCV hw2-2')
parser.add_argument('-d', '--data_path', default='./HW2/', type=str)
parser.add_argument('-o', '--output_path', default='./output_image/', type=str)
args = parser.parse_args()


def color_segmentation(img):
    img_flat = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    kmeans = KMeans(n_clusters=10, max_iter=1000).fit(img_flat)
    seg = np.reshape(kmeans.labels_, (img.shape[0], img.shape[1]))
    return label2rgb(seg)


def texture_segmentation(img, filter, img_color=None):
    for i in range(filter.shape[-1]):
        if i == 0:
            img_flat = convolve2d(img, filter[:,:,i], mode='same', boundary='symm').reshape(img.shape[0]*img.shape[1], 1)
        else:
            conv = convolve2d(img, filter[:,:,i], mode='same', boundary='symm').reshape(img.shape[0]*img.shape[1], 1)
            img_flat = np.hstack((img_flat, conv))
    if img_color is not None:
        color_feature = np.reshape(img_color, (img_color.shape[0]*img_color.shape[1], 3))
        img_flat = np.hstack((img_flat, color_feature))
    kmeans = KMeans(n_clusters=6, max_iter=1000).fit(img_flat)
    seg = np.reshape(kmeans.labels_, (img.shape[0], img.shape[1]))
    return label2rgb(seg)


if __name__ == '__main__':

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ### Load image and filter ###

    img_zebra = io.imread(os.path.join(args.data_path, './Problem2/zebra.jpg'))
    img_mount = io.imread(os.path.join(args.data_path, './Problem2/mountain.jpg'))
    mat = loadmat(os.path.join(args.data_path, './Problem2/filterBank.mat'))
    filter = mat['F']


    ### (a) Color Segmentation ###

    seg_zebra = color_segmentation(img_zebra)
    seg_mount = color_segmentation(img_mount)
    io.imsave(os.path.join(args.output_path, 'zebra_CS.jpg'), seg_zebra)
    io.imsave(os.path.join(args.output_path, 'mountain_CS.jpg'), seg_mount)

    seg_lab_zebra = color_segmentation(rgb2lab(img_zebra))
    seg_lab_mount = color_segmentation(rgb2lab(img_mount))
    io.imsave(os.path.join(args.output_path, 'zebra_CS_lab.jpg'), seg_lab_zebra)
    io.imsave(os.path.join(args.output_path, 'mountain_CS_lab.jpg'), seg_lab_mount)


    ### (b) Texture Segmentation ###

    seg_zebra = texture_segmentation(rgb2gray(img_zebra), filter, None)
    seg_mount = texture_segmentation(rgb2gray(img_mount), filter, None)
    io.imsave(os.path.join(args.output_path, 'zebra_TS_t.jpg'), seg_zebra)
    io.imsave(os.path.join(args.output_path, 'mountain_TS_t.jpg'), seg_mount)

    seg_lab_zebra = texture_segmentation(rgb2gray(img_zebra), filter, rgb2lab(img_zebra))
    seg_lab_mount = texture_segmentation(rgb2gray(img_mount), filter, rgb2lab(img_mount))
    io.imsave(os.path.join(args.output_path, 'zebra_TS_tc.jpg'), seg_lab_zebra)
    io.imsave(os.path.join(args.output_path, 'mountain_TS_tc.jpg'), seg_lab_mount)
