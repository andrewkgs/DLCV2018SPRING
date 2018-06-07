from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

cnn = np.load('cnn_feature.npy')
rnn = np.load('rnn_feature.npy')
gt = np.genfromtxt('gt.txt')

embed_cnn = TSNE(n_components=2, random_state=1).fit_transform(cnn)
embed_rnn = TSNE(n_components=2, random_state=1).fit_transform(rnn)

#print(embed_cnn.shape)
#print(embed_rnn.shape)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(embed_cnn[:, 0], embed_cnn[:, 1], c=gt, cmap='hsv')
plt.title('CNN-based feature')
plt.subplot(122)
plt.scatter(embed_rnn[:, 0], embed_rnn[:, 1], c=gt, cmap='hsv')
plt.title('RNN-based feature')
plt.tight_layout()
plt.savefig('feat.png')
#plt.show()



