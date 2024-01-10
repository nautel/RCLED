import argparse
import datetime
from pathlib import Path
import torch
#from load_data import load_data
import os
import numpy as np
import torch.utils.data as DATA
from tqdm import tqdm
import re
#from mscred import MSCRED
from model import RCLED
import time

import utils

def get_args_parser():
    parser = argparse.ArgumentParser('RCLED', add_help=False)

    parser.add_argument("--stacked_windows", default=(10, 30, 60), type=int, nargs='+',
                        help="Specifying the windows for stacked for dataset")
    parser.add_argument("--dataset", default='synthetic', choices=['synthetic', 'SMAP', 'sensorted', 'MSL'])
    # model parameters
    parser.add_argument('--patch_size', default=5, type=int)

    parser.add_argument('--out_dim')

    # Training/Optimization parameters
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=0.0002, type=int)
    parser.add_argument("--optimizer", default="adam", type=str,
                        choices=['adam', 'sgd', 'lars'])
    parser.add_argument("--error_limit", default=0.0001, help="""The limitation of c1 and c2 for early stopping algorithm""")
    parser.add_argument("--batch_size_per_gpu", default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    #Misc
    parser.add_argument('--data_dir', default='./data/data_matrix/', type=str,
                        help="""Please specify path to the data.""")
    parser.add_argument('--output_dir', default="./result/",
                        help='Path to save logs and checkpoints.')
    parser.add_argument('saveckp_freq', default=5, type=int,
                        help='Save checkpoint every x epochs.')

    return parser
device = "cuda" if torch.cuda.is_available else "cpu"
#############


 #   def Robust_Train(dataloader, model, optimizer, epoch, device_name, lamda, verbose=True, error=0.0001):
def train_model(args):

    # ===================== preparing data ... =====================
    data = []
    if args.dataset == 'synthetic':
        FILE_NAME = list()
        for window in range(args.stacked_windows):
            name = f'{args.noise_level}_win{window}.npy'
            FILE_NAME.append(name)


    for file in FILE_NAME:
        PATH = os.path.join(args.input_dir, args.dataset, file)
        matrix = np.load(PATH)
        data.append(matrix)
        dataset = torch.from_numpy(np.array(data)).float()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.bath_size_per_gpu,
        shuffle=True
    )

    print(f"Data loaded: there are {len(dataset)} signature matrices")
    # ============ building networks ============
    robust_model = RCLED()
    # ============ preparing loss ============
    rcled_loss = ?
    # ============ preparing optimizer ============
    params_groups = robust_model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    # ============ init schedulers ============
    # ============ optional resume training ============
    start_time = time.time()
    print("Starting RCLED training !")
    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of RCLED ... ============
        train_stats = train_one_epoch(robust_model, rcled_loss, data_loader, optimizer, epoch, args)
        # ============ writing logs ... ============
        save_dict = {'model': robust_model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch + 1,
                     'args': args,
                     'loss': rcled_loss.state_dict()}
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, rcled_loss, data_loader,
                    optimizer, epoch, args):



s = torch.zeros([5, 3, 30, 30])
    ls = torch.zeros([5, 3, 30, 30])
    model = model.to(device_name)
    for epoch in range(epoch):
        train_loss_sum, n = 0.0, 0
        for x in tqdm(dataloader):
            x = x.squeeze()

            x = x.to(device_name)
            s = s.to(device_name)
            lt = ls.to(device_name)
            ld = x - s

            model.train()
            loss = torch.mean((model(ld) - ld[-1].unsqueeze(0)) ** 2)
            train_loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            ld_t = model(ld)
            ld = trans_one2five(ld_t)

            s = x - ld
            s = s.detach().cpu().numpy()

            mu = s.size / (4.0 * np.linalg.norm(s.reshape(-1, s.shape[-1] * s.shape[-1]), 1))
            s = shrink(lamda / mu, s.reshape(-1)).reshape(x.shape)
            #np.save(f'C:/Users/C19 - Admin/PycharmProjects/rmscred_v4/data_rmscred/sensor_tech/noise_train/s_lam{lam}.npy',s)
            np.save(f'C:/Users/C19 - Admin/PycharmProjects/rmscred_v4/data_rmscred/sensor_tech/noise_train/s_syn_lam{lamda}.npy', s)
            XF_norm = torch.norm(x, 'fro')

            s = torch.from_numpy(s)
            s = s.to(device_name)
            ld = ld.to(device_name)
            x = x.to(device_name)
            XF_norm = XF_norm.to(device_name)

            c1 = torch.norm((x - ld - s), 'fro') / XF_norm
            c2 = np.min([mu, np.sqrt(mu)]) * torch.norm(ls - ld - s) / XF_norm

            ls = ld + s
            if c1 < error and c2 < error:
                print("early break")
                continue

            n += 1
            # print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))
        if verbose:
            print("c1 :", c1)
            print("c2 :", c2)

        print("[Epoch %d/%d] [loss: %f]" % (epoch + 1, epoch, train_l_sum / n))

def train_noise(dataloader, model, optimizer, epochs, device_name, lam, noise_factor, verbose=True, error=0.0001):
    model = model.to(device_name)
    s = torch.zeros([5, 3, 30, 30])
    ls = torch.zeros([5, 3, 30, 30])
    for epoch in range(epochs):
        train_l_sum, n = 0.0, 0
        for x in tqdm(dataloader):
            x = x.to(device_name)
            x = x.squeeze()
            s = s.to(device_name)
            ls = ls.to(device_name)

            ld = x - s
            # print(type(x))
            model.train()
            loss = torch.mean((model(ld) - ld[-1].unsqueeze(0)) ** 2)
            train_l_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            ld_t = model(ld)
            ld = trans_one2five(ld_t)

            s = x - ld

            s = s.detach().cpu().numpy()

            mu = s.size / (4.0 * np.linalg.norm(s.reshape(-1, s.shape[-1] * s.shape[-1]), 1))

            s = shrink(lam / mu, s.reshape(-1)).reshape(x.shape)
            #np.save(f'C:/Users/C19 - Admin/PycharmProjects/rmscred_v4/data_rmscred/sensor_tech/noise_train/s_lam{lam}.npy',s)
            np.save(f'C:/Users/C19 - Admin/PycharmProjects/rmscred_v4/data_rmscred/sensor_tech/noise_train/s_syn_lam{lam}_noise{noise_factor}.npy',s)
            XF_norm = torch.norm(x, 'fro')
            s = torch.from_numpy(s)
            s = s.to(device_name)
            ld = ld.to(device_name)
            x = x.to(device_name)
            XF_norm = XF_norm.to(device_name)

            c1 = torch.norm((x - ld - s), 'fro') / XF_norm
            c2 = np.min([mu, np.sqrt(mu)]) * torch.norm(ls - ld - s) / XF_norm

            ls = ld + s
            if c1 < error and c2 < error:
                print("early break")
                continue

            n += 1
            # print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))
        if verbose:
            print("c1 :", c1)
            print("c2 :", c2)

        print("[Epoch %d/%d] [loss: %f]" % (epoch + 1, epochs, train_l_sum / n))

def test(dataloader, model, reconstructed_path):
    index = 11
    loss_list = []
    model = model.to(device)
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            x = x.squeeze()
            reconstructed_matrix = model(x)
            path_temp = os.path.join(reconstructed_path, 'reconstructed_data_' + str(index) + ".npy")
            np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            # l = criterion(reconstructed_matrix, x[-1].unsqueeze(0)).mean()
            # loss_list.append(l)
            # print("[test_index %d] [loss: %f]" % (index, l.item()))
            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RCLED', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parent=True, exst_ok=True)
    train_model(args)

    epochs = 20
    noise_factors = [0, 1, 2, 3, 4, 5]
    lams = [100, 10, 1, 0.1, 0.01, 0.001]

    for noise_factor in noise_factors:
        for lam in lams:
            print("------training on {}-------".format(device))
            print(f"Noise: {noise_factor}")
            print(f"Lams: {lam}")
            train_path = f'./data_rmscred/noise_factor_0{noise_factor}/train/'
            #train_path = f'./data_rmscred/sensor_tech/train/'
            save_model = 'models/RCLED/'
            loader = load_data(train_path, shuffle=True)

            mscred_model = MSCRED(3, 256)

            adam = torch.optim.Adam(mscred_model.parameters(), lr=0.0002)
            #train(loader, mscred_model, adam, epochs, device, lam=lam)
            train_noise(loader, mscred_model, adam, epochs, device, lam=lam, noise_factor=noise_factor)

            #model_names = f'RCLED_sensor_tech_surface_data_lam{lam}.pth'
            model_names = f'RCLED_syn_data_lam{lam}_noise{noise_factor}.pth'
            torch.save(mscred_model.state_dict(), save_model + model_names)
"""
i