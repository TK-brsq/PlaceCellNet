import numpy as np
import random
from typing import List, Tuple


#CNN\training_and_testing_data\mnist_images_SDR_classifiers_training.npy

class DataLoader:
    def __init__(self, dir: str, data_size_p_c: int, grid_size: int = 5, train: bool = True):
        self.dir = dir
        self.train = train
        str_train = 'training.npy' if train else 'testing.npy'

        self.vectors = np.load(dir + '/mnist_SDRs_SDR_classifiers_' + str_train)
        #print(f'vectors {self.vectors.shape}')
        self.labels = np.load(dir + '/mnist_labels_SDR_classifiers_' + str_train)

        self.patch_count = self.vectors.shape[1] // 128
        self.vectors = np.reshape(self.vectors, (-1, 128, grid_size, grid_size)) # datasize, dim, w, h
        self.vectors = np.transpose(self.vectors, axes=(0,1,3,2)) # datasize, dim, h, w
        self.vectors = np.reshape(self.vectors, (-1, 128, grid_size**2))
        self.vectors = np.transpose(self.vectors, (0, 2, 1))
        #print(f'vectors {self.vectors.shape}')
        self.classes = len(set(self.labels))

        self.all_data = []
        self.all_labels = []

        self.set_dataset(data_size_p_c)

        pass

    def __iter__(self):
        for data, target in zip(self.all_data, self.all_labels):
            yield data, target

    def set_dataset(self, data_size: int):
        all_data = []
        all_labels = []

        for label in range(self.classes):
            arg_label = [i for i, v in enumerate(self.labels) if v == label]
            arg_label = arg_label[:data_size]
            assert len(arg_label) == data_size, f'Data size is lower than {data_size} in class {label}'

            vectors = self.vectors[arg_label]
            #print(f'vectors : {vectors.shape}')
            #one = vectors[0,0,:]
            #i = np.where(one > 0)[0]
            #print(f'one sdr : {len(i)}')
            labels = self.labels[arg_label]
            patch_indices = data_size * [list(range(self.patch_count))]
            data = [[v, p] for v, p in zip(vectors, patch_indices)]
            #print(f'len(data) : {len(data)}')

            all_data.extend(data)
            all_labels.extend(labels)
        
        # shuffle
        combined = list(zip(all_data, all_labels))
        random.shuffle(combined)

        list1_shuffled, list2_shuffled = zip(*combined)
        s_data = list(list1_shuffled)
        s_labels = list(list2_shuffled)

        self.all_data = s_data
        self.all_labels = s_labels

class PseudoData:
    def __init__(self, patch_size=4, num_patterns=5, data_size=2):
        self.ps = patch_size
        self.nump = num_patterns
        self.data_size = data_size

        self.all_labels = [i for i in range(data_size)]
        
        patterns = self.make_pattern()
        self.all_datas = self.make_data(patterns)

        pass

    def make_data(self, patterns: List[np.ndarray]):
        images = []
        for img_idx in range(self.data_size):
            vecs = []
            patchs = [i for i in range(self.ps)]
            # 右下パッチ以外は for文
            for pat_idx in range(self.ps - 1):
                vec = patterns[pat_idx]
                vecs.append(vec)
            # 右下パッチは 画像番号によってパターンを変える
            vec = patterns[self.ps-1+img_idx]
            vecs.append(vec)
            images.append([vecs, patchs])

        return images


    def make_pattern(self):
        patterns = []
        for i in range(self.nump):
            a = np.zeros((128), dtype=int)
            idx = np.random.choice(128, 29, replace=False)
            print(np.sort(idx))
            a[idx] = 1
            patterns.append(a)

        return patterns

    def __iter__(self):
        for data, target in zip(self.all_datas, self.all_labels):
            yield data, target

# __main__
if __name__ == '__main__':
    dataloader = DataLoader('dataset', 1)
    #dataloader = PseudoData()

    for datas, labels in dataloader:
        print(f'vec : {datas[0][2]}')
        print(f'patch : {datas[1]}')
        print(f'labels : {labels}')
        break
# vec =data[0], patch = data[1]
