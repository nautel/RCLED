import datetime
import torch
from tqdm import tqdm
import time
from loss import *
from dataset import *


def shrink(epsilon, X_in):
    x = X_in.clone().detach()
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
        config=config,
        phase='train'
    )
    print('print(train_dataset.shape):', train_dataset.shape)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True
    )

    # ===================== preparing optimizer =====================
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate
    )
    # ===================== preparing loss function =================
    criterion = nn.MSELoss()
    # ===================== preparing checkpoints =====================

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    # ===================== training =====================
    train_losses = []

    start_time = time.time()
    print("Starting RCLED training !")
    for epoch in range(config.model.epoch):
        train_loss = 0.
        for step, X in enumerate(tqdm(data_loader)):
#            print('X.shape:', X.shape)
#            print('step:', step)
            L = torch.zeros(X.shape)
            S = torch.zeros(X.shape)

            X = X.to(config.model.device)
            L = L.to(config.model.device)
            S = S.to(config.model.device)

            mu = torch.numel(X) / (4.0 * torch.norm(X, 1))
            XFnorm = torch.norm(X, 'fro')
            LS0 = L + S

            for i in range(config.model.inner_iter):
                L = X - S
                ############# fit autoencoder ##########
                model.train()
                output = model(L)
                loss = criterion(output[-1], L[-1])
                optimizer.zero_grad()
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                ########################################
                model.eval()
                L = model(L)
                # alternating project, now project to S
                S = shrink(config.hyperparameter.lamda / mu, (X - L).reshape(X.numel())).reshape(X.size())
                ########################################
                c1 = torch.norm(X - L - S, 'fro') / XFnorm
                c2 = torch.min(mu, torch.sqrt(mu)) * torch.norm(LS0 - L - S) / XFnorm
                if c1 < config.hyperparameter.error_limit and c2 < config.hyperparameter.error_limit:
                    print("Next data sample")
                    continue
                ## save
                LS0 = L + S
              

            save_path = os.path.join(config.model.result_dir, f'{config.data.name}', 'train')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, f'Epoch{epoch}_Step{step}_S'), L.detach().cpu().numpy())
            np.save(os.path.join(save_path, f'Epoch{epoch}_Step{step}_L'), S.detach().cpu().numpy())

        with torch.no_grad():
            # logging
            if (epoch + 1) % 1 == 0 :
                print(f"Epoch {epoch + 1} | Loss: {train_loss/len(data_loader)}")
            if (epoch + 1) % 10 == 0 and epoch > 0 :
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.name, category)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch + 1)+'.pt'))

        train_losses.append(train_loss)
#        print("shrink parameter:", config.hyperparameter.lamda / mu)
#        print("X shape: ", X.shape)
#        print("L shape: ", L.shape)
#        print("S shape: ", S.shape)
#        print("mu: ", mu)
#        print("XFnorm: ", XFnorm)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
