import numpy as np
import matplotlib.pyplot as plt

pre = np.genfromtxt('./prediction/OP01-R03-BaconAndEggs.txt')
gt = np.genfromtxt('./gt/OP01-R03-BaconAndEggs.txt')

plt.figure(figsize=(20, 4))
plt.scatter(np.arange(pre.shape[0]), np.zeros(pre.shape[0]), c=pre, cmap='hsv')
plt.scatter(np.arange(gt.shape[0]), np.ones(gt.shape[0]), c=gt, cmap='hsv')
plt.savefig('./frame_label.png')

