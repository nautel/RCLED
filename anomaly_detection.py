from numpy import True_
import torch
from visualization import *
from metrics import *
from tqdm import tqdm
import pandas as pd
import os 
import re
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"


def anomaly_score(x, x0, config):
    matrix_error = torch.square(torch.sub(x, x0))
    number_broken = len(matrix_error[matrix_error > config.hyperparameters.gama])
    return number_broken

def get_label(config):
    if config.data.name == 'SMAP':
        PATH = os.path.join(config.data.data_label, f'{config.data.name}.csv')
        labeled_anomalies = pd.read_csv(PATH)
        category = labeled_anomalies[labeled_anomalies['chan_id'] == config.data.category]
        length = category['num_values'].item()
#        print('length', length)
        anomaly_sequences = [int(s)//config.signature_matrix.time_step for s in re.findall(r'\b\d+\b', category['anomaly_sequences'].item())]
#        print('anomaly_sequences', anomaly_sequences)
        length = length // config.signature_matrix.time_step

        begin = anomaly_sequences[::2]
        end = anomaly_sequences[1::2]
        labels = np.zeros(length)
        for i in range(len(begin)):
            labels[begin[i]:end[i]] = 1   
     
    return labels

class Anomaly_Detection():
    def __init__(self, model, config) -> None:
        self.valid_dataset = DatasetMaker(root=config.data.data_dir,
                                          config=config,
                                          phase='valid')
        self.validloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=True,
        )
        self.test_dataset = DatasetMaker(root=config.data.data_dir,
                                         config=config,
                                         phase='test')

        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=True,
        )

        self.model = model
        self.config = config

        self.labels = get_label(config)

    def __call__(self) -> any:

        valid_score = []
        test_score = []
        predictions = []

        with torch.no_grad():
            for it, x in tqdm(enumerate(self.validloader)):
               # print('validation')
                x = x.to(self.config.model.device)
                x0 = self.model(x)
                score = anomaly_score(x, x0, self.config)
                valid_score.append(score)

            rho = max(valid_score) * self.config.hyperparameters.beta

            for it, x in tqdm(enumerate(self.testloader)):
                x = x.to(self.config.model.device)
                x0 = self.model(x)
                score = anomaly_score(x, x0, self.config)
                if score > rho:
                    predictions.append(1)
                else:
                    predictions.append(0)
                test_score.append(score)
              
            OUT_PATH = os.path.join(self.config.model.result_dir, self.config.data.name ,'test')
            print(np.array(test_score))
            print(OUT_PATH + f'/AnomalyScore{self.config.data.category}.npy')


            np.save(OUT_PATH + f'/AnomalyScore{self.config.data.category}.npy', np.array(test_score))
#        print(predictions)


#        metric = Metric(self.labels, predictions, self.config)
#        metric.optimal_threshold()
#        if self.config.metric.auroc:
#            print('AUROC: ({:.1f})'.format(metric.auroc() * 100))
#        if self.config.metric.pro:
#            print('PRO: {:.1f}'.format(metric.pro() * 100))
#        if self.config.metrics.misclassifications:
#            metric.misclassified()#

#        if not os.path.exists('results'):
#            os.mkdir('reuslts')
#        if self.config.metrics.visualisation:
#            visualize(labels_list, predictions)
