import os

import torch

from dataset import *
from visualize import *
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"

class Anomaly_Detection:
    def __init__(self, model, config) -> None:
        self.test_dataset = DatasetMaker(root=config.data.data_dir,
                                         category=config.data.category,
                                         config=config,
                                         is_train=False)

        self.testloader = torch.utils.data.DataLoader(
            self.testloader,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=False,

        )

        self.model = model
        self.config = config

    def __call__(self) -> Any:

        with torch.no_grad():
            for input, labels in self.testloader:
                input = input.to(self.config.model.device)
                x0 = self.model(input)
                score = anomaly_score(x0, input, self.config)





