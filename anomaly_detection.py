from numpy import True_
import torch
from visualization import *
from metrics import *
from tqdm import tqdm
import pandas as pd
import os 
import re
import numpy as np
from sklearn.metrics import roc_auc_score
from dataset import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"


def anomaly_score(x, x0, config):
    matrix_error = torch.square(torch.sub(x, x0))
    number_broken = len(matrix_error[matrix_error > config.hyperparameter.gama])
    #print(matrix_error.sum().item())
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
    if config.data.name =='synthetic':
        anomalies_labels = pd.read_csv('/content/RCLED/data/labeled_anomalies/synthetic.csv')
        labels = np.zeros(10000)
        begin_ids = (anomalies_labels['begin_id'] - 10000).values 
        lengths = (anomalies_labels['length']).values
        for i in range(len(begin_ids)):
          index = begin_ids[i]
          length = lengths[i]
          labels[index : index + length] = 1
    return labels

class Anomaly_Detection():
    def __init__(self, model, config) -> None:
        valid_timeseries = DatasetMaker(root=config.data.data_dir,
                                          config=config,
                                          phase='valid')
        self.valid_dataset = MyDataset(valid_timeseries)
        self.validloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=True,
        )


        test_timeseries = DatasetMaker(root=config.data.data_dir,
                                         config=config,
                                         phase='test')
    
        self.test_dataset = MyDataset(test_timeseries)

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
        self.labels = self.labels[: :config.signature_matrix.time_step]


    def __call__(self) -> any:

        valid_score = []
        test_score = []
        predictions = []

        with torch.no_grad():
            print('Validation!')
            for it, batch in tqdm(enumerate(self.validloader)):
                x = batch[0]
               # print('validation')
                x = x.to(self.config.model.device)
                x0 = self.model(x)
                score = anomaly_score(x, x0, self.config)
                valid_score.append(score)
            #print(valid_score)
            rho = max(valid_score) * self.config.hyperparameter.beta
            print('Test!')
            for it, batch in tqdm(enumerate(self.testloader)):
                x = batch[0]
                x = x.to(self.config.model.device)
                x0 = self.model(x)
                score = anomaly_score(x, x0, self.config)
                if score > rho:
                    predictions.append(1)
                else:
                    predictions.append(0)
                test_score.append(score)
            
            OUT_PATH = os.path.join(self.config.model.result_dir, self.config.data.name ,'test')
            #print('test_score', test_score)
            #print('len test_score', len(test_score))
            #print('label', self.labels)
            #print('len lables', len(self.labels[4:]))
            print('ROC_AUC:', roc_auc_score(self.labels[4:], test_score))
            #print(OUT_PATH + f'/AnomalyScore{self.config.data.category}.npy')
            #np.save(OUT_PATH + f'/testAnomalyScore{self.config.data.category}.npy', np.array(test_score))
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
