from load_data import load_data
import os
import torch
import numpy as np
import torch.utils.data as DATA
from tqdm import tqdm
import re
from mscred import MSCRED
import time

device = "cuda" if torch.cuda.is_available else "cpu"
#############

def shrink(epsilon, input):
    x = input.copy()
    t1 = x > epsilon
    t2 = x < epsilon
    t3 = x > -epsilon
    t4 = x < -epsilon
    x[t2 & t3] = 0
    x[t1] = x[t1] - epsilon
    x[t4] = x[t4] + epsilon
    return x

def trans_one2five(x_in):
    x_1 = torch.roll(x_in, 1, dims=0)
    x_2 = torch.roll(x_in, 2, dims=0)
    x_3 = torch.roll(x_in, 3, dims=0)
    x_4 = torch.roll(x_in, 4, dims=0)
    return torch.cat((x_4, x_3, x_2, x_1, x_in), 0)


def Robust_Train(dataloader, model, optimizer, epoch, device_name, lamda, verbose=True, error=0.0001):
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