import numpy as np
from skimage import io


def mean_face(X, shape):
    mean = np.mean(X, axis=0)
    io.imsave('./output_image/mean_face.png', mean.reshape(shape).astype(np.uint8))


def eigenface(X, shape, num):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)

    for i in range(num):
        eigen = U[:, i]
        eigen -= np.min(eigen)
        eigen /= np.max(eigen)
        eigen *= 255
        save_name = './output_image/eigenface' + str(i+1) + '.png'
        io.imsave(save_name, eigen.reshape(shape).astype(np.uint8))


def reconstruct(X, X_target, shape, num):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)

    X_target_center = X_target - mean
    weight = np.dot(X_target_center, U[:, :num])

    recon = mean + np.dot(weight, U[:, :num].T)
    print('MSE of reconstructed image for n = %d : %.6f' %(num, MSE(X_target, recon)))
    save_name = './output_image/reconstruct' + str(num)+ '.png'
    io.imsave(save_name, recon.reshape(shape).astype(np.uint8))


def MSE(X_org, X_rec):
    diff = np.abs(X_org - X_rec)
    return np.sum((diff ** 2)) / X_org.shape[0]


if __name__ == "__main__":

    ### Load image data ###

    train_set, test_set = [], []
    for i in range(40):
        for j in range(10):
            file_name = "./hw1_dataset/" + str(i+1) + "_" + str(j+1) + ".png"
            image = io.imread(file_name)
            if j < 6:
                train_set.append(image.flatten())
            else:
                test_set.append(image.flatten())

    train_set = np.array(train_set)
    test_set = np.array(test_set)


    ### Generate mean face of training set ###

    mean_face(train_set, (56, 46))


    ### Generate the first three eigenfaces ###

    eigenface(train_set, (56, 46), 3)


    ### Generate reconstructed images ###

    for n in [3, 50, 100, 239]:
        reconstruct(train_set, train_set[0], (56, 46), n)


    ### Apply 3-fold cross validation ###
