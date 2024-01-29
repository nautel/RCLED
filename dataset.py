import os
import numpy as np
import torch


def DatasetMaker(root, config):
    # ===================== preparing data ... =====================
    data = []
    if config.data.name == 'NoiseLevel20':
        FILE_NAME = list()
        for window in [10, 30, 60]:
            name = f'NoiseLevel{config.synthetic.noise_level}_Window{window}.npy'
            FILE_NAME.append(name)

    for name in FILE_NAME:
        PATH = os.path.join(root, 'NoiseLevel20', name)
        matrix = np.load(PATH)
        data.append(matrix)
        dataset = torch.from_numpy(np.array(data)).float()

    dataset = np.transpose(dataset, (1, 0, 2, 3))
    return dataset
