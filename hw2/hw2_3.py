from cv2 import imread, imwrite, xfeatures2d, drawKeypoints
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import os
import argparse
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser(description='DLCV hw2-3')
parser.add_argument('-d', '--data_path', default='./HW2/', type=str)
parser.add_argument('-o', '--output_path', default='./output_image/', type=str)
args = parser.parse_args()


def InterestPoints(dir, H_threshold):
    surf = xfeatures2d.SURF_create(H_threshold)
    first = True
    for category in ['Coast', 'Forest', 'Highway', 'Mountain', 'Suburb']:
        img_name = os.listdir(os.path.join(dir, category))
        for file in img_name:
            img_path = os.path.join(os.path.join(dir, category), file)
            img = imread(img_path, 0)
            _, descriptors = surf.detectAndCompute(img, None)
            #print(len(descriptors))
            descriptors = descriptors[:200]
            if descriptors is None:
                print('No descriptor in {}'.format(img_path))
                continue
            if first:
                interest_pts = descriptors
                first = False
            else:
                interest_pts = np.append(interest_pts, descriptors, axis=0)
    print('interest points shape: {}'.format(interest_pts.shape))
    return interest_pts


def VisualWords(interest_pts, n_clusters, max_iter):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(interest_pts)
    print('visual words shape: {}'.format(kmeans.cluster_centers_.shape))   
    return kmeans


def PCR(interest_pts, kmeans):
    idx = 0
    select_idx = []
    select_tar = []
    select_pts = []
    for label in kmeans.labels_:
        if label < 6:
            for num in range(6):
                if label == num:
                    select_idx.append(idx)      
                    select_tar.append(label)
                    select_pts.append(interest_pts[label])
        idx += 1
    select_idx = np.array(select_idx)
    select_tar = np.array(select_tar)
    select_pts = np.array(select_pts)
    print('select_idx shape: {}'.format(select_idx.shape))
    print('select_tar shape: {}'.format(select_tar.shape))
    print('select_pts shape: {}'.format(select_pts.shape))

    select_center = kmeans.cluster_centers_[:6,:]
    print('select center shape: {}'.format(select_center.shape))

    pca = PCA(n_components=3).fit(interest_pts)
    X = pca.transform(interest_pts)
    print('X shape: {}'.format(X.shape))

    select_X = X[select_idx, :]
    print('select X shape: {}'.format(select_X.shape))

    select_center_X = pca.transform(select_center)
    print('select center X shape: {}'.format(select_center_X.shape))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(select_X[:,0], select_X[:,1], select_X[:,2], c=select_tar, s=2, marker=".")
    ax.scatter(select_center_X[:,0], select_center_X[:,1], select_center_X[:,2], c=[0, 1, 2, 3, 4, 5], s=30, edgecolors="r")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA subspace')
    #plt.show()
    plt.clf()


def HardSum(descriptors, VW):
    BoW_array = np.zeros(VW.shape[0])
    BoW_feature = np.zeros(VW.shape[0])
    for i in descriptors:
        for j in range(VW.shape[0]):
            BoW_feature[j] = euclidean(i, VW[j])
        BoW_array[np.argmin(BoW_feature)] += 1
    BoW_array /= np.sum(BoW_array)
    return BoW_array


def SoftSum(descriptors, VW):
    BoW_array = np.zeros(VW.shape[0])
    BoW_feature = np.zeros(VW.shape[0])
    for i in descriptors:
        for j in range(VW.shape[0]):
            BoW_feature[j] = euclidean(i, VW[j])
        factor = np.sum(np.reciprocal(BoW_feature))
        BoW_feature = 1.0 / (factor*BoW_feature)
        BoW_array += BoW_feature
    BoW_array /= np.sum(BoW_array)
    return BoW_array


def SoftMax(descriptors, VW):
    BoW_array = np.zeros(VW.shape[0])
    BoW_feature = np.zeros(VW.shape[0])
    for i in descriptors:
        for j in range(VW.shape[0]):
            BoW_feature[j] = euclidean(i, VW[j])
        factor = np.sum(np.reciprocal(BoW_feature))
        BoW_feature = 1.0 / (factor*BoW_feature)
        for k in range(BoW_array.shape[0]):
            if BoW_feature[k] > BoW_array[k]:
                BoW_array[k] = BoW_feature[k]
    return BoW_array


def BoW(dir, VW, strategy, plot, H_threshold):
    surf = xfeatures2d.SURF_create(H_threshold)
    BoW_whole = []
    for category in ['Coast', 'Forest', 'Highway', 'Mountain', 'Suburb']:
        img_name = os.listdir(dir + category)
        for file in img_name:
            img_path = dir + category + '/' + file
            img = imread(img_path, 0)
            _, descriptors = surf.detectAndCompute(img, None)
            descriptors = descriptors[:100]
            if strategy == 'Hard-Sum':
                BoW_image = HardSum(descriptors, VW)
            elif strategy == 'Soft-Sum':
                BoW_image = SoftSum(descriptors, VW)
            elif strategy == 'Soft-Max':
                BoW_image = SoftMax(descriptors, VW)
            BoW_whole.append(BoW_image)
    BoW_whole = np.array(BoW_whole)
    print('BoW shape: {}'.format(BoW_whole.shape))

    if plot is True:
        for idx in [0, 10, 20, 30, 40]:
            plt.bar(np.arange(BoW_whole.shape[1]), BoW_whole[idx], width=1, edgecolor='b')
            #plt.ylim((0.0, 1.0))
            plt.title('BoW by {}'.format(strategy))
            plt.xlabel('')
            plt.ylabel('')
            plt.savefig(os.path.join(args.output_path, 'BoW_by_{}_{}'.format(strategy, idx)))
            #plt.show()
            plt.clf()

    return BoW_whole


if __name__ == '__main__':

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ### (a) ###
    img = imread(args.data_path +'/Problem3/train-10/Coast/image_0006.jpg', 0) # 0 refers to gray scale
    surf = xfeatures2d.SURF_create(1250)
    keypoints, descriptors = surf.detectAndCompute(img, None)
    print('number of keypoints: {}'.format(len(keypoints)))

    img_det = drawKeypoints(img, keypoints, None, (0,0,255))
    imwrite(os.path.join(args.output_path, 'kp.jpg'), img_det)



    ### (b),(c),(d)-(i) ###
    H_threshold = 400
    n_clusters = 50
    max_iter = 5000
    train_path = os.path.join(args.data_path, 'Problem3/train-10/')
    test_path = os.path.join(args.data_path, 'Problem3/test-100/')

    interest_pts = InterestPoints(train_path, H_threshold)
    kmeans = VisualWords(interest_pts, n_clusters=n_clusters, max_iter=max_iter)
    PCR(interest_pts, kmeans)

    BoW_HS = BoW(train_path, kmeans.cluster_centers_, strategy='Hard-Sum', plot=True, H_threshold=H_threshold)
    BoW_SS = BoW(train_path, kmeans.cluster_centers_, strategy='Soft-Sum', plot=True, H_threshold=H_threshold)
    BoW_SM = BoW(train_path, kmeans.cluster_centers_, strategy='Soft-Max', plot=True, H_threshold=H_threshold)

    BoW_HS_test = BoW(test_path, kmeans.cluster_centers_, strategy='Hard-Sum', plot=False, H_threshold=H_threshold)
    BoW_SS_test = BoW(test_path, kmeans.cluster_centers_, strategy='Soft-Sum', plot=False, H_threshold=H_threshold)
    BoW_SM_test = BoW(test_path, kmeans.cluster_centers_, strategy='Soft-Max', plot=False, H_threshold=H_threshold)

    label_train = np.zeros(50)
    label_test = np.zeros(500)
    for i in range(5):
        for j in range(10):
            label_train[10*i+j] = i
        for k in range(100):
            label_test[100*i+k] = i

    BoW_HS, BoW_SS, BoW_SM, label_train = shuffle(BoW_HS, BoW_SS, BoW_SM, label_train, random_state=0)

    clf_HS = KNeighborsClassifier(n_neighbors=5)
    clf_HS.fit(BoW_HS, label_train)
    #print(clf_HS.predict(BoW_HS_test))
    test_score = clf_HS.score(BoW_HS_test, label_test)
    print('Hard-Sum score: {}'.format(test_score))

    clf_SS = KNeighborsClassifier(n_neighbors=5)
    clf_SS.fit(BoW_SS, label_train)
    #print(clf_SS.predict(BoW_SS_test))
    test_score = clf_SS.score(BoW_SS_test, label_test)
    print('Soft-Sum score: {}'.format(test_score))

    clf_SM = KNeighborsClassifier(n_neighbors=5)
    clf_SM.fit(BoW_SM, label_train)
    #print(clf_SM.predict(BoW_SM_test))
    test_score = clf_SM.score(BoW_SM_test, label_test)
    print('Soft-Max score: {}'.format(test_score))
 

    ### (d)-(ii) ###
    H_threshold = 400
    n_clusters = 50
    max_iter = 5000
    train_path = os.path.join(args.data_path, 'Problem3/train-100/')
    test_path = os.path.join(args.data_path, 'Problem3/test-100/')

    interest_pts = InterestPoints(train_path, H_threshold)
    kmeans = VisualWords(interest_pts, n_clusters=n_clusters, max_iter=max_iter)
    #PCR(interest_pts, kmeans)

    BoW_HS = BoW(train_path, kmeans.cluster_centers_, strategy='Hard-Sum', plot=True, H_threshold=H_threshold)
    BoW_SS = BoW(train_path, kmeans.cluster_centers_, strategy='Soft-Sum', plot=True, H_threshold=H_threshold)
    BoW_SM = BoW(train_path, kmeans.cluster_centers_, strategy='Soft-Max', plot=True, H_threshold=H_threshold)

    BoW_HS_test = BoW(test_path, kmeans.cluster_centers_, strategy='Hard-Sum', plot=False, H_threshold=H_threshold)
    BoW_SS_test = BoW(test_path, kmeans.cluster_centers_, strategy='Soft-Sum', plot=False, H_threshold=H_threshold)
    BoW_SM_test = BoW(test_path, kmeans.cluster_centers_, strategy='Soft-Max', plot=False, H_threshold=H_threshold)

    label_train = np.zeros(500)
    label_test = np.zeros(500)
    for i in range(5):
        for j in range(100):
            label_train[100*i+j] = i
            label_test[100*i+j] = i            

    BoW_HS, BoW_SS, BoW_SM, label_train = shuffle(BoW_HS, BoW_SS, BoW_SM, label_train, random_state=0)

    clf_HS = KNeighborsClassifier(n_neighbors=5)
    clf_HS.fit(BoW_HS, label_train)
    #print(clf_HS.predict(BoW_HS_test))
    test_score = clf_HS.score(BoW_HS_test, label_test)
    print('Hard-Sum score: {}'.format(test_score))

    clf_SS = KNeighborsClassifier(n_neighbors=5)
    clf_SS.fit(BoW_SS, label_train)
    #print(clf_SS.predict(BoW_SS_test))
    test_score = clf_SS.score(BoW_SS_test, label_test)
    print('Soft-Sum score: {}'.format(test_score))

    clf_SM = KNeighborsClassifier(n_neighbors=5)
    clf_SM.fit(BoW_SM, label_train)
    #print(clf_SM.predict(BoW_SM_test))
    test_score = clf_SM.score(BoW_SM_test, label_test)
    print('Soft-Max score: {}'.format(test_score))

