import numpy as np
from typing import  Dict, List
from dataloader import DataLoader
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Weight:
    def __init__(self, column_count: int, cell_count: int, dendrite_count: int, input_dim: int):
        
        self.column_count = column_count
        self.cell_count = cell_count
        self.dendrite_count = dendrite_count
        self.input_dim = input_dim

        initial_W = np.random.normal(0.45, 0.1, (column_count, cell_count, dendrite_count, input_dim)).astype(np.float16)
        self.W = np.clip(initial_W, 0, 1)

        self.active_dendrites = np.zeros((column_count, cell_count, dendrite_count), dtype=bool)
        self.fired_cells = np.zeros((column_count, cell_count), dtype=bool)

        pass

    def get_connected_synapses(self, threshold: float = 0.5):
        return self.W > threshold
    
    def get_active_dendrites_and_cells(self, column_idx: int, x, threshold: int = 1):

        # Step1 : calculate activation and activate dendrites
        #activations = self.get_connected_synapses()[column_idx, :, :, :] @ x # activations.shape = (cell, dendrite)
        activations = self.W[column_idx, :, :, :] @ x # activations.shape = (cell, dendrite)
        #print(f'act of dend {activations}')
        #active_dendrites = activations >= threshold # active_dendrites.shape = (cell, dendrite)
        
        # Step2 : select best dendrite
        active_dendrites = np.zeros((self.cell_count, self.dendrite_count), dtype=bool)
        for i in range(self.cell_count):
            max_idx = np.argmax(activations[i, :])
            active_dendrites[i, :] = False
            active_dendrites[i, max_idx] = True
        
        self.active_dendrites[column_idx] = active_dendrites
        #print(f'active dendrites: {self.active_dendrites}')
        
        # この一塊は改善の余地あり
        self.fired_cells.fill(False)
        '''
        fired_cells = np.zeros((self.column_count, self.cell_count), dtype=bool)
        if np.max(activations) >= threshold:
            max_act_in_cell = np.max(activations, axis=1) # shape = (cell)
            winner_cell_idx = np.argmax(max_act_in_cell)
            fired_cells[column_idx, winner_cell_idx] = True
        self.fired_cells = fired_cells
        #print(f'fired cells: {fired_cells}')
        '''
        winner_cell_coord = np.unravel_index(np.argmax(activations), (self.column_count, self.cell_count))
        self.fired_cells[column_idx, winner_cell_coord[1]] = True

        return active_dendrites, winner_cell_coord

    def update_weights(self, column_idx: int, x, sigma: float = 1.0, lr = 0.05):
        # Step 1 : coordinate of winner cell
        fc_idx = np.where(self.fired_cells[column_idx, :] == True)[0]
        wd_idx = np.where(self.active_dendrites[column_idx, fc_idx, :] == True)[1]
        #print(f'idx: {fc_idx}, {wd_idx}')

        # Step 2 : calculate distance and influence
        location = np.arange(self.cell_count)
        distance = abs(location - int(fc_idx))
        influence = np.exp(-(distance**2) / (2 * (sigma**2)))

        # Step 3 : update weights
        self.W[column_idx, :, :, :] += lr * influence[:, np.newaxis, np.newaxis] * (x - self.W[column_idx, :, :, :])


        # winnerセルはinputに近づく
        '''
        self.W[column_idx, fc_idx, wd_idx, :] += 2 * lr * (x - self.W[column_idx, fc_idx, wd_idx, :])
        # loserセルはinputから遠ざかる
        self.W[column_idx, :, :, :] -= lr * (x - self.W[column_idx, :, :, :])
        self.W[column_idx, :, :, :] = np.clip(self.W[column_idx, :, :, :], 0, 1)
        '''
        return self.W
    
    def fix_weights(self, column_idx, cell_idx, dest, new_weight: float = 0.51):
        w = np.zeros(self.W[column_idx, cell_idx, 0, :].shape)
        #print(f'w.shape : {w.shape}')
        w[dest] = new_weight
        self.W[column_idx, cell_idx, 0, :] = w
        pass

class CorticalNeuralNetwork:
    def __init__(self):
        self.column_count = 25
        self.cell_count = 32
        self.dendrite_count = 1
        self.input_dim = 128

        self.lateral_W = Weight(25, 32, 1, 25*32)
        self.basal_W = Weight(25, 32, 1, 128)
        pass

    def train_basal(self, column_idx: int, x : np.ndarray):
        _, coordinate = self.basal_W.get_active_dendrites_and_cells(column_idx, x)
        self.basal_W.update_weights(column_idx, x)
        #print(coordinate)

        return coordinate

    def train_image(self, patterns: List[np.ndarray]):
        act_cell_record = []
        for patch_idx, patch in enumerate(patterns):
            act_cell_coord = self.train_basal(patch_idx, patch)
            act_cell_record.append(act_cell_coord)
        
        self.train_lateral(act_cell_record)

        return act_cell_record[8][1]
    
    def train_lateral(self, act_cells: List[int]):
        for i in range(len(act_cells)):
            dest_coord = act_cells[:i] + act_cells[i+1:]
            #print(dest_coord)
            dest_gbidx = [x*self.cell_count + y for x, y in dest_coord]
            #print(dest_gbidx)
            #print(f'column_idx: {column_idx}, cell_idx: {cell_idx}, dest: {dest}')
            self.lateral_W.fix_weights(act_cells[i][0], act_cells[i][1], dest_gbidx)

    def eval():
        pass


#print(f'W : {w.W}')
#con = w.get_connected_synapses()
#print(np.sum(con))
#print(f'diff: {w_after - w_before}')
#print(w.W)

cnn = CorticalNeuralNetwork()
loader = DataLoader('dataset', 30)

#d = [{i: 0} for i in range(10)]
d = [0 for _ in range(10)]
dist_of_class_in_cell = [d.copy() for _ in range(32)]
#print(dist_of_class_in_cell[0])
#print(dist_of_class_in_cell)

for data, label in tqdm(loader): # data: [vectors, patch_indices], label: int
    vec = data[0]
    #print(f'vec.shape: {vec.shape}')
    act_cell = cnn.train_image(vec)
    #print(f'act_cell: {act_cell}, label: {label}')
    dist_of_class_in_cell[act_cell][label] += 1

map = np.array(dist_of_class_in_cell)
plt.imshow(map)
plt.colorbar(label='value')
plt.show()

    



"""
Sprint Backlog

Weight implementation
    L dendrite
    L cell
    L update, fix_update
"""



"""
Design

class Weight:
    def init():
    # 本当はどっちかだけ
        self.W_lateral = np.array((25, 32, 32, 25*32), float16)
        self.W_basal = np.array((25, 32, 32, 128), float16)


    def activate_dendrite()


    def activate_cell(active_column)
        activations = self.W_basal[active_column, :, :] @  input_vector
        fired_cells = activations > threshold
        return fired_cells
        


    def get_connected():
        return self.W > threshold

class CorticalNN:
    def __init__(self):
        self.lateral_W = Weight()
        self.basal_W = Weight()

    def train_image()
        act_cell_record = []
        for patch in image:
            act = self.train()
            act_cell_record.append(act)
        
        self.train_lateral(act_cell_record)


    def eval_image()

    def train_basal()
       act_from_basal = self.basal_W.activate_cell(column_idx)
       self.basal_W.update_weights(column_idx, act_from_basal)
       return act_from_basal
    
    def train_lateral(act_cells)
        self.lateral_w.update_weights(act_cells)

    def eval()

class experiment()
    def init():
        self.model = Model()

    def train()
        for all_images:
            model.train_image()

    def eval()
"""