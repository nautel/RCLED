import os
import numpy as np
import torch


def DatasetMaker(root, config, phase):
    # ===================== preparing data ... =====================
    data = []
    train_percent = 0.9
    if config.data.name == 'synthetic':
        if phase == 'train' or phase == 'valid':
            FILE_NAME = list()
            for window in [10, 30, 60]:
                name = f'NoiseLevel{config.synthetic.noise_level}_Window{window}.npy'
                FILE_NAME.append(name)

            for name in FILE_NAME:
                PATH = os.path.join(root, config.data.name, 'train', name)
                matrix = np.load(PATH)
                data.append(matrix)
                dataset = torch.from_numpy(np.array(data)).float()
                
            dataset = np.transpose(dataset, (1, 0, 2, 3))
            train_length = int((dataset.shape[0]) * train_percent)

            if phase == 'train':
                dataset = dataset[:train_length, :, :, :]
            if phase == 'valid':
                dataset = dataset[train_length:, :, :, :]

        if phase == 'test':
            FILE_NAME = list()
            for window in [10, 30, 60]:
                name = f'NoiseLevel{config.synthetic.noise_level}_Window{window}.npy'
                FILE_NAME.append(name)

            for name in FILE_NAME:
                PATH = os.path.join(root, config.data.name, 'test', name)
                matrix = np.load(PATH)
                data.append(matrix)
                dataset = torch.from_numpy(np.array(data)).float()

            dataset = np.transpose(dataset, (1, 0, 2, 3))

    if config.data.name == 'SMAP':
        if phase == 'train' or phase == 'valid':
            FILE_NAME = list()
            for window in [10, 30, 60]:
                name = f'{config.data.category}_Window{window}.npy'
                FILE_NAME.append(name)

            for name in FILE_NAME:
                PATH = os.path.join(root, config.data.name, 'train', name)
                matrix = np.load(PATH)
                data.append(matrix)
                dataset = torch.from_numpy(np.array(data)).float()
                
            dataset = np.transpose(dataset, (1, 0, 2, 3))
            train_length = int((dataset.shape[0]) * train_percent)

            if phase == 'train':
                dataset = dataset[:train_length, :, :, :]
            if phase == 'valid':
                dataset = dataset[train_length:, :, :, :]

        if phase == 'test':
            FILE_NAME = list()
            for window in [10, 30, 60]:
                name = f'{config.data.category}_Window{window}.npy'
                FILE_NAME.append(name)

            for name in FILE_NAME:
                PATH = os.path.join(root, config.data.name, 'test', name)
                matrix = np.load(PATH)
                data.append(matrix)
                dataset = torch.from_numpy(np.array(data)).float()

            dataset = np.transpose(dataset, (1, 0, 2, 3))

    return dataset
