from dataloader import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

W = np.random.normal(0.45, 0.1, (32, 128)).astype(np.float16)
W = np.clip(W, 0, 1)

def wc(W, input):
    act = W @ input
    #print(act.shape)
    wci = np.argmax(act)
    return wci

loader = DataLoader('dataset', 30)
map = np.zeros((32, 10), dtype=int)
for data, label in tqdm(loader): # data: [vectors, patch_indices], label: int
    patch_idx = 0
    vec = data[0][patch_idx]
    wci = wc(W, vec)
    map[wci, label] += 1

plt.imshow(map)
plt.colorbar(label='value')
plt.show()
