import datetime
import torch
import os
import numpy as np
from tqdm import tqdm
import time
import tqdm as tqdm
from loss import *
from dataset import *


def shrink(epsilon, X_in):
    x = X_in.copy()
    t1 = x > epsilon
    t2 = x < epsilon
    t3 = x > -epsilon
    t4 = x < -epsilon
    x[t2 & t3] = 0
    x[t1] = x[t1] - epsilon
    x[t4] = x[t4] + epsilon
    return x


def trainer(model, category, config):
    """
    Train model

    :param model: the RCLED model
    :param category: the category of the dataset
    """
    # ===================== preparing data ... =====================
    train_dataset = DatasetMaker(
        root=config.data.data_dir,
        config=config)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True
    )

    # ===================== preparing optimizer =====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    # ===================== preparing checkpoints =====================

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    # ===================== training =====================

    start_time = time.time()
    print("Starting RCLED training !")
    for epoch in range(config.model.epoch):
        for step, X in tqdm(enumerate(data_loader)):
            L = torch.zeros(X.shape)
            S = torch.zeros(X.shape)
            mu = torch.numel(X) / (4.0 * torch.norm(X, 1))
            XFnorm = torch.norm(X, 'fro')
            print("shrink parameter:", config.hyperparameter.lamda / mu)
            print("X shape: ", X.shape)
            print("L shape: ", L.shape)
            print("S shape: ", S.shape)
            print("mu: ", mu)
            print("XFnorm: ", XFnorm)
            LS0 = L + S

            for i in range(config.model.inner_iter):
                ######train_one_epoch
                L = X - S
                ############# fit autoencoder ##########
                loss = get_loss(model, L)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ########################################
                model.eval()
                L = model(L)
                # alternating project, now project to S
                S = shrink(config.hyperparameter.lamda / mu, (X - L).reshape(X.size)).reshape(X.shape)
                ########################################
                c1 = torch.norm(X - L - S, 'fro') / XFnorm
                c2 = np.min([mu, np.sqrt(mu)]) * torch.norm(LS0 - L - S) / XFnorm
                if c1 < config.hyperparameter.error_limit and c2 < config.hyperparameter.error_limit:
                    print("Next data sample")
                    continue
                ## save
                LS0 = L + S

            save_path = os.path.join(config.model.result_dir, f'{config.data.name}', 'train')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, f'Epoch{epoch}_S'), L.detach().cpu().numpy())
            np.save(os.path.join(save_path, f'Epoch{epoch}_L'), S.detach().cpu().numpy())

            # logging
            if (epoch + 1) % 2 == 0 and step == 0:
                print(f"Epoch {epoch + 1} | Loss: {loss.item()}")
            if (epoch + 1) % 10 == 0 and epoch > 0 and step == 0:
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch + 1)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))