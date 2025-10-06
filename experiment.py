from TopBottomCNN import WLayerCNN
from dataloader import DataLoader, PseudoData
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm


class Experiment:
    def __init__(self, train_dataset=None,
                 test_dataset=None,
                 datasize_p_c: int = 1,
                 dir: str = 'dataset',
                 eval_on_train: bool = False):
        
        self.model = WLayerCNN(128, 32, 29, 10)

        if train_dataset is not None:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
        else:
            self.train_dataset = DataLoader(dir, datasize_p_c)
            self.test_dataset = self.train_dataset

        pass

    def train(self):
        for data, target in tqdm(self.train_dataset):
            #print(f'target : {target}')
            vectors = data[0]
            #print(f'len_vectors : {len(vectors[0])}')
            patch_indices = data[1]
            #print(f'len_patch : {len(patch_indices)}')
            self.model.learn_image(vectors, patch_indices, target)
        
        w = self.model.classifier.weight_matrix
        #print(w[:,32])
        #print(w[:,160])

        pass


    def infer(self):
        correct_dist_all = []
        for data, target in tqdm(self.test_dataset):
            #if target > 0:
            #    break
            vectors = data[0]
            pred = self.model.infer_image(vectors)

            correct = pred[:, target]
            correct_dist_all.append(correct)
        
        return correct_dist_all



# __main__
if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)
    #train_set = PseudoData()
    #test_set = PseudoData()
    experiment = Experiment(train_dataset=None, test_dataset=None, datasize_p_c=10, eval_on_train=True)

    print('train')
    experiment.train()
    print('infer')
    result = experiment.infer()

    now = datetime.now()
    str_now = now.strftime("%m-%d-%H-%M-%S")
    np.save('results/' + str_now + '.npy', result)

    print('finish')