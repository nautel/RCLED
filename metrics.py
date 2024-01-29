import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

class Metric:
    def __init__(self, labels_list, predictions, config) -> None:
        self.labels_list = labels_list
        self.predictions = predictions

    def auroc(self):
        auroc = roc_auc_score(self.labels_list, self.predictions)
        return auroc

    def pre_rec(self):
        pre_rec = precision_recall_curve(self.labels_list, self.predictions)
        return pre_rec

    def optimal_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)

        # calculate youden's J statistic for each thresold
        youden_j = tpr - fpr
        optimal_threshold_index = np.argmax(youden_j)
        optimal_thresold = thresholds[optimal_threshold_index]
        return optimal_thresold

    def misclassified(self):
        predictions = torch.tensor(self.predictions)
        labels_list = torch.tensor(self.labels_list)
        predictions0_1 = (predictions > self.threshold).int()
        for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
            print('Sample : ', i, ' predicted as: ',p.item() ,' label is: ',l.item(),'\n' ) if l != p else None