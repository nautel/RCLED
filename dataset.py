import os
import numpy as np
import torch


def DatasetMaker(root, config):
    # ===================== preparing data ... =====================
    data = []
    if config.data.name == 'synthetic':
        FILE_NAME = list()
        for window in [10, 30, 60]:
            name = f'NoiseLevel{config.synthetic.noise_level}_Window{window}.npy'
            FILE_NAME.append(name)

    for file in FILE_NAME:
        PATH = os.path.join(root, 'synthetic', file)
        matrix = np.load(PATH)
        data.append(matrix)
        dataset = torch.from_numpy(np.array(data)).float()

    return dataset
