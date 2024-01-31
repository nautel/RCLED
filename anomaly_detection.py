import torch
from visualization import *
from metrics import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"


def anomaly_score(x, x0, config):
    matrix_error = torch.square(torch.sub(x, x0))
    number_broken = len(matrix_error[matrix_error > config.hyperparameters.gama])
    return number_broken


class Anomaly_Detection():
    def __init__(self, model, config) -> None:
        self.valid_dataset = DatasetMaker(root=config.data.data_dir,
                                          config=config,
                                          phase='valid')
        self.validloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=False,
        )
        self.test_dataset = DatasetMaker(root=config.data.data_dir,
                                         config=config,
                                         phase='test')

        self.testloader = torch.utils.data.DataLoader(
            self.testloader,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=False,
        )

        self.model = model
        self.config = config

    def __call__(self) -> any:

        valid_score = []
        labels_list = []
        predictions = []

        with torch.no_grad():

            for x, label in self.validloader:
                x.to(self.config.model.device)
                x0 = self.model(x)
                score = anomaly_score(x, x0, self.config)
                valid_score.append(score.item())

            rho = max(valid_score) * self.config.hyperparameter.beta

            for x, label in self.testloader:
                x.to(self.config.model.device)
                x0 = self.model(x)
                score = anomaly_score(x, x0, self.config)

                labels_list.append(label)
                if score > rho:
                    predictions.append(1)
                else:
                    predictions.append(0)

        metric = Metric(labels_list, predictions, self.config)
        metric.optimal_threshold()
        if self.config.metric.auroc:
            print('AUROC: ({:.1f})'.format(metric.auroc() * 100))
        if self.config.metric.pro:
            print('PRO: {:.1f}'.format(metric.pro() * 100))
        if self.config.metrics.misclassifications:
            metric.misclassified()

        if not os.path.exists('results'):
            os.mkdir('reuslts')
        if self.config.metrics.visualisation:
            visualize(labels_list, predictions)
