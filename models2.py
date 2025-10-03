import numpy as np
from typing import  Dict, List

class Weight:
    def __init__(self, column_count: int, cell_count: int, dendrite_count: int, input_dim: int):
        
        self.W = np.random.normal(0.5, 0.1, (column_count, cell_count, dendrite_count, input_dim)).astype(np.float16)
        pass

    def get_connected_synapses(self, threshold: float = 0.5):
        return self.W > threshold
    
    def get_active_dendrites(self, column_idx: int, x, threshold: int = 10):
        activations = self.W[column_idx, :, :, :] @ x
        active_dendrites = activations > threshold
        return active_dendrites

    
    def get_fired_cells(self, column_idx: int, x, threshold: int = 10):
        activations = self.W[column_idx, :, :, :] @ x
        fired_cells = activations > threshold
        return fired_cells
    
    def update_weights(self, column_idx: int, fired_cell, winner_dendrite, x, lr = 0.01):
        # winnerセルはinputに近づく
        self.W[column_idx, fired_cell, winner_dendrite, :] += 2 * lr * (x - self.W[column_idx, fired_cell, winner_dendrite, :])

        # loserセルはinputから遠ざかる
        loser_cells = np.logical_not(fired_cell) 
        self.W[column_idx, :, :, :] -= lr * (x - self.W[column_idx, loser_cells, :, :])

        self.W[column_idx, :, :, :] = np.clip(self.W[column_idx, loser_cells, :, :], 0, 1)

w = Weight(1, 1, 1, 128)
print(w.W)
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