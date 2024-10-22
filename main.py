import argparse
import os.path
from numpy import False_, format_float_scientific
import torch
import pandas as pd
from omegaconf import OmegaConf
from generating_syn_dataset import *
from generating_signature_matrix import *
from model import RCLEDmodel
from train import trainer
from anomaly_detection import Anomaly_Detection

def build_model(config):
    if config.data.name == "synthetic":
        model = RCLEDmodel(num_vars=30, in_channels_ENCODER=3, in_channels_DECODER=256)
    if config.data.name == "SMAP":
        model = RCLEDmodel(num_vars=25, in_channels_ENCODER=3, in_channels_DECODER=256)
    if config.data.name == "MSL":
        model = RCLEDmodel(num_vars=55, in_channels_ENCODER=3, in_channels_DECODER=256)
    return model


def parse_args():
    parser = argparse.ArgumentParser('RCLED')
    parser.add_argument('-cfg', '--config',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
                        help='config file')
    parser.add_argument('--preparing',
                        default=False,
                        help='Preparing data')
    parser.add_argument('--train',
                        default=False,
                        help='Train the robust model')
    parser.add_argument('--detection',
                        default=True,
                        help='Detection anomalies')
    args, unknowns = parser.parse_known_args()
    return args


def synthetic(config):
    np.random.seed(42)
    print(f"Generating {config.data.name} time series")
    if not os.path.exists(config.synthetic.output_dir):
        os.makedirs(config.synthetic.output_dir)

    anomalies = pd.read_csv(os.path.join(config.data.data_label, 'synthetic.csv'))
    s = generate_time_series_dataset(config.synthetic.num_vars, config.synthetic.ts_lengths,
                                     config.synthetic.noise_level)
    s = adding_anomaly(s, anomalies)
    # train
    np.save(config.synthetic.output_dir + 'train' + f'/NoiseLevel{config.synthetic.noise_level}', s[:, :10000])
    # test
    np.save(config.synthetic.output_dir + 'test' + f'/NoiseLevel{config.synthetic.noise_level}', s[:, 10000:])


def preparing(config):
    np.random.seed(42)
    print("Generating signature matrix from time series")
    #    if not os.path.exists(os.path.join(config.signature_matrix.input_dir, config.data.name)):
    #       os.makedirs(os.path.join(config.signature_matrix.input_dir, config.data.name))
    for phase in ['train', 'test']:
        if not os.path.exists(os.path.join(config.signature_matrix.output_dir, config.data.name, phase)):
            os.makedirs(os.path.join(config.signature_matrix.output_dir, config.data.name, phase))

        if config.data.name == 'synthetic':
            PATH = os.path.join(config.signature_matrix.input_dir, config.data.name, phase,
                                f'{config.data.category}.npy')
            data = np.load(PATH)
            data_normalized = normalization(data)
            for window in config.signature_matrix.windows:
                matrix = ts2matrix(data_normalized, window, config.signature_matrix.time_step)
                SAVE_PATH = os.path.join(config.signature_matrix.output_dir, config.data.name, phase,
                                        f'{config.data.category}_Window{window}.npy')
                np.save(SAVE_PATH, matrix)

        if config.data.name == 'SMAP':  
            INPUT_PATH = os.path.join(config.signature_matrix.input_dir, config.data.name, phase,
                                f'{config.data.category}.npy')
            data = np.load(INPUT_PATH)
            data = data.transpose()
            print('Series SHAPE: ', data.shape)
            data_normalized = normalization(data)
            for window in config.signature_matrix.windows:
                matrix = ts2matrix(data_normalized, window, config.signature_matrix.time_step)
                SAVE_PATH = os.path.join(config.signature_matrix.output_dir, config.data.name, phase,
                                  f'{config.data.category}_Window{window}.npy')
                np.save(SAVE_PATH, matrix)

def train(config):
    torch.manual_seed(42)
    np.random.seed(42)
    model = build_model(config)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = model.to(config.model.device)
    model.train()
    # sao chép và phân phối mô hình vào nhiều GPU nếu có
    trainer(model, config.data.category, config)


def detection(config):
    model = build_model(config)
    checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.name, config.data.category,
                            str(config.model.load_checkpoint)+'.pt'))
    model.load_state_dict(checkpoint)
    model.to(config.model.device)
    model.eval()
    predict_anomalies = Anomaly_Detection(model, config)
    predict_anomalies()


if __name__ == "__main__":
    # torch.cuda.empty_cache()
    args = parse_args()
    config = OmegaConf.load(args.config)
    print("Datasets: ", config.data.name)
    # torch.manuel_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.preparing:
        print('Preparing ...')
        if config.data.name == 'synthetic':
            synthetic(config)
            preparing(config)
        if config.data.name == 'SMAP':
            preparing(config)
    if args.train:
        print('Training ...')
        train(config)
    if args.detection:
        print('Detecting ...')
        detection(config)
