import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsClassifier


output_path = './output_image/'
shape = (56, 46)


def mean_face(X, shape, save):
    mean = np.mean(X, axis=0)
    if save:
        save_name = 'mean_face.png'
        io.imsave(output_path + save_name, mean.reshape(shape).astype(np.uint8))
        print('< {} saved >\n'.format(save_name))


def eigen_face(X, shape, eigen_num, save):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)

    for i in range(eigen_num):
        eigen = U[:, i]
        if save:
            eigen -= np.min(eigen)
            eigen /= np.max(eigen)
            eigen *= 255
            save_name = 'eigen_face_' + str(i+1) + '.png'
            io.imsave(output_path + save_name, eigen.reshape(shape).astype(np.uint8))
            print('< {} saved >\n'.format(save_name))


def reconstruct(X, X_target, shape, eigen_num, save):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)

    X_target_center = X_target - mean
    weight = np.dot(X_target_center, U[:, :eigen_num])

    if save:
        recon = (mean + np.dot(weight, U[:, :eigen_num].T))
        print('MSE of reconstructed image for n = {} : {}'.format(eigen_num, MSE(X_target, recon)))
        save_name = 'reconstruct_' + str(eigen_num)+ '.png'
        io.imsave(output_path + save_name, recon.reshape(shape).astype(np.uint8))
        print('< {} saved >\n'.format(save_name))
    return weight


def MSE(X_org, X_rec):
    diff = np.abs(X_org - X_rec)
    return (diff ** 2).mean()


def train_and_valid(image_f1, image_f2, image_f3, label_f1, label_f2, label_f3, clf_train, n):

    X_t, X_v = [], []
    image_train = np.concatenate((image_f2, image_f3), axis=0)
    image_valid = image_f1
    for img in image_train:
        X_t.append(reconstruct(image_train, img, shape, n, False))
    for img in image_valid:
        X_v.append(reconstruct(image_train, img, shape, n, False))
    X_t, X_v = np.array(X_t), np.array(X_v)
    Y_t, Y_v = np.concatenate((label_f2, label_f3), axis=0), label_f1
    clf.fit(X_t, Y_t)
    score_v1 = clf.score(X_v, Y_v)
    print('score_v1: {}'.format(score_v1))

    X_t, X_v = [], []
    image_train = np.concatenate((image_f1, image_f3), axis=0)
    image_valid = image_f2
    for img in image_train:
        X_t.append(reconstruct(image_train, img, shape, n, False))
    for img in image_valid:
        X_v.append(reconstruct(image_train, img, shape, n, False))
    X_t, X_v = np.array(X_t), np.array(X_v)
    Y_t, Y_v = np.concatenate((label_f1, label_f3), axis=0), label_f2
    clf_train.fit(X_t, Y_t)
    score_v2 = clf_train.score(X_v, Y_v)
    print('score_v2: {}'.format(score_v2))

    X_t, X_v = [], []
    image_train = np.concatenate((image_f1, image_f2), axis=0)
    image_valid = image_f3
    for img in image_train:
        X_t.append(reconstruct(image_train, img, shape, n, False))
    for img in image_valid:
        X_v.append(reconstruct(image_train, img, shape, n, False))
    X_t, X_v = np.array(X_t), np.array(X_v)
    Y_t, Y_v = np.concatenate((label_f1, label_f2), axis=0), label_f3
    clf.fit(X_t, Y_t)
    score_v3 = clf.score(X_v, Y_v)
    print('score_v3: {}'.format(score_v3))

    print('average validation accuracy: {}\n'.format((score_v1+score_v2+score_v3)/3))


def train(image_train, label_train, image_test, label_test, clf_test, n):
    X_train, X_test = [], []
    for img in image_train:
        X_train.append(reconstruct(image_train, img, shape, n, False))
    for img in image_test:
        X_test.append(reconstruct(image_train, img, shape, n, False))
    X_train, X_test = np.array(X_train), np.array(X_test)
    Y_train, Y_test = label_train, label_test
    clf_test.fit(X_train, Y_train)
    test_score = clf_test.score(X_test, Y_test)
    print('test score: {}'.format(test_score))


if __name__ == "__main__":

    ### Load image data ###

    train_image, test_image = [], []
    train_label, test_label = [], []
    for i in range(40):
        for j in range(10):
            file_name = "./hw1_dataset/" + str(i+1) + "_" + str(j+1) + ".png"
            image = io.imread(file_name)
            if j < 6:
                train_image.append(image.flatten())
                train_label.append(i+1)
            else:
                test_image.append(image.flatten())
                test_label.append(i+1)

    train_image, train_label = np.array(train_image), np.array(train_label)
    test_image, test_label = np.array(test_image), np.array(test_label)


    ### Generate mean face of training set ###

    mean_face(train_image, shape, True)


    ### Generate the first three eigen faces ###

    eigen_face(train_image, shape, 3, True)


    ### Generate reconstructed images ###

    for n in [3, 50, 100, 239]:
        reconstruct(train_image, train_image[0], shape, n, True)


    ### Apply 3-fold cross validation ###

    image_f1, image_f2, image_f3 = [], [], []
    Y_f1, Y_f2, Y_f3 = [], [], []
    for i in range(len(train_image)//3):
        image_f1.append(train_image[3*i])
        image_f2.append(train_image[3*i+1])
        image_f3.append(train_image[3*i+2])
        Y_f1.append(train_label[3*i])
        Y_f2.append(train_label[3*i+1])
        Y_f3.append(train_label[3*i+2])

    image_f1, Y_f1 = np.array(image_f1), np.array(Y_f1)
    image_f2, Y_f2 = np.array(image_f2), np.array(Y_f2)
    image_f3, Y_f3 = np.array(image_f3), np.array(Y_f3)

    for k in [1, 3, 5]:
        clf = KNeighborsClassifier(n_neighbors=k)
        for n in [3, 50, 159]:
            print('k = {}, n = {}'.format(k, n))
            train_and_valid(image_f1, image_f2, image_f3, Y_f1, Y_f2, Y_f3, clf, n)


    ### Apply the hyperparameter choice on test set ###

    k_test, n_test = 1, 50
    clf= KNeighborsClassifier(n_neighbors=k_test)
    train(train_image, train_label, test_image, test_label, clf, n_test)
