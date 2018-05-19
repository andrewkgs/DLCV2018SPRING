import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def SetArgument():
    parser = argparse.ArgumentParser(description='Plot learning curve.')
    parser.add_argument('-o', '--output_path', default='./output/', type=str)
    return parser.parse_args()

if __name__ == '__main__':

    args = SetArgument()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    log_1_1 = np.genfromtxt('vae_logs/log_1_KLD.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    log_1_2 = np.genfromtxt('vae_logs/log_1_MSE.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    log_2_1 = np.genfromtxt('dcgan_logs/log_2_G.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    log_2_2 = np.genfromtxt('dcgan_logs/log_2_D.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    log_2_3 = np.genfromtxt('dcgan_logs/log_2_D_r.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    log_2_4 = np.genfromtxt('dcgan_logs/log_2_D_f.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    #log_3_1 = np.genfromtxt('acgan_logs/log_3_.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    #log_3_2 = np.genfromtxt('acgan_logs/log_3_.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    #log_3_3 = np.genfromtxt('acgan_logs/log_3_.csv', delimiter=',', dtype=np.float32)[1:, 1:]
    #log_3_4 = np.genfromtxt('acgan_logs/log_3_.csv', delimiter=',', dtype=np.float32)[1:, 1:]

    ##########

    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plt.plot(log_1_1[:, 0], log_1_1[:, 1])
    plt.title('KLD loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.plot(log_1_2[:, 0], log_1_2[:, 1])
    plt.title('MSE loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'fig1_2.jpg'))
    plt.clf()

    ##########

    plt.subplot(121)
    plt.plot(log_2_1[:, 0], log_2_1[:, 1], 'b', label='generator')
    plt.plot(log_2_2[:, 0], log_2_2[:, 1], 'r', label='discriminator')
    plt.title('Discriminator/Generator loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(loc=0)

    plt.subplot(122)
    plt.plot(log_2_3[:, 0], log_2_3[:, 1], 'g', label='real image')
    plt.plot(log_2_4[:, 0], log_2_4[:, 1], 'm', label='fake image')
    plt.title('Discriminator real/fake image loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(loc=0)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'fig2_2.jpg'))

    ##########
    '''
    plt.subplot(121)
    plt.plot(log_3_1[:, 0], log_3_1[:, 1], label='discriminator')
    plt.plot(log_3_2[:, 0], log_3_2[:, 1], label='generator')
    plt.title('Discriminator/Generator loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(loc=0)

    plt.subplot(122)
    plt.plot(log_3_3[:, 0], log_3_3[:, 1], label='real image')
    plt.plot(log_3_4[:, 0], log_3_4[:, 1], label='fake image')
    plt.title('Discriminator real/fake image loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(loc=0)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'fig3_2.jpg'))
    '''
