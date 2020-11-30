import os
import torch
import argparse
import numpy as np
from train import Trainer
from test import Evaluation
from network.lstm import Net
from network.conv_gru import NickNet
from network.dense_net import DenseNet

from preprocessing import FileManager, DataManager

obj_cuda = torch.cuda.is_available()


def get_args():
    parser = argparse.ArgumentParser()

    # Specify the desired WAV-format
    parser.add_argument('--sample_rate', default=16000, type=int)
    parser.add_argument('--sample_channels', default=1, type=int)
    parser.add_argument('--sample_width', default=2, type=int)

    # Name of folder to save the data files in
    parser.add_argument('--data_folder', default="data", type=str)

    # min/max length for slicing the voice files
    parser.add_argument('--slice_min_ms', default=1000, type=int)
    parser.add_argument('--slice_max_ms', default=5000, type=int)

    # frame size to use for the labelling
    parser.add_argument('--frame_size_ms', default=30, type=int)

    # prepare_audio should always be false unless one wants to force a new data generator
    parser.add_argument('--prepare_audio', default=False, type=bool)
    # train_model will enforce a fresh training of our defined models, rathre than loading them from local storage.
    parser.add_argument('--train_model', default=True, type=bool)

    # Hyper-parameters
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--frames', default=30, type=int)
    parser.add_argument('--features', default=24, type=int)
    parser.add_argument('--step_size', default=6, type=int)
    return parser.parse_args()


def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if obj_cuda:
        torch.cuda.manual_seed_all(seed)


def create_folder(path):
    directory = ['images', 'models', 'data']
    for i in directory:
        if not os.path.exists(os.getcwd() + "/" + i):
            os.makedirs(os.getcwd() + "/" + i)


def prepare_audio(args):
    print("Loading Files ...")
    speech_dataset = FileManager(args, "speech", "processing_data/korean-data")
    noise_dataset = FileManager(args, "noise", "processing_data/QUT-NOISE")

    speech_dataset.prepare_files()
    noise_dataset.prepare_files(normalize=True)

    print("Collecting frames ...")

    speech_dataset.collect_frames()
    noise_dataset.collect_frames()

    print("Labelling frames ...")
    speech_dataset.label_frames()

    print("set up for use in neural networks")
    data_manager = DataManager(args)
    data = data_manager.prepare_data(speech_dataset, noise_dataset)
    return data


def net_path(epoch, title):
    part = os.getcwd() + '/models/' + title + '/' + title
    if epoch >= 0:
        return part + '_epoch' + str(epoch).zfill(3) + '.net'
    else:
        return part + '.net'


def load_net(epoch=14, title='net'):
    if obj_cuda:
        return torch.load(net_path(epoch, title))
    else:
        return torch.load(net_path(epoch, title), map_location='cpu')


def load_model(args):
    if args.train_model:
        trainer = Trainer(args)
        # LSTM, small, γ = 0
        set_seed()
        net = Net(args, large=False)
        # trainer.train(net, data, title='net', gamma=0)

        # LSTM, large, γ = 2
        set_seed()
        net_large = Net(args)
        # trainer.train(net_large, data, title='net_large', gamma=2)

        # Conv + GRU, small, γ = 2
        set_seed()
        gru = NickNet(args, large=False)
        # trainer.train(gru, data, title='gru', gamma=2)

        # Conv + GRU, large, γ = 2
        set_seed()
        gru_large = NickNet(args)
        trainer.train(gru_large, data, title='gru_large', gamma=2)

        # DenseNet, small, γ = 2
        set_seed()
        densenet = DenseNet(large=False, batch_size=args.batch_size)
        # trainer.train(densenet, data, title='densenet', use_adam=False, lr=1, momentum=0.7, gamma=2)

        # DenseNet, large, γ = 2
        set_seed()
        densenet_large = DenseNet(large=True, batch_size=args.batch_size)
        # trainer.train(densenet, data, title='densenet_large', use_adam=False, lr=1, momentum=0.7, gamma=2)
    else:
        net = load_net(title='net')
        net_large = load_net(title='net_large')
        gru = load_net(title='gru')
        gru_large = load_net(title='gru_large')
        densenet = load_net(title='densenet')
        densenet_large = load_net(title='densenet_large')
    return net, net_large, gru, gru_large, densenet, densenet_large


if __name__ == '__main__':
    args = get_args()
    create_folder("./")  # 현재폴더에 필요한 파일 생성
    data = prepare_audio(args)
    net, net_large, gru, gru_large, densenet, densenet_large = load_model(args)
    print('gru: ')
    print(gru)
    print('gru large : ')
    print(gru_large)

    networks = {
        'RNN': net,
        # 'RNN (large)': net_large,
        # 'Conv + RNN': gru,
        'Conv + RNN (large)': gru_large,
        # 'DenseNet': densenet,
        # 'DenseNet (large)': densenet_large
    }
    evalator = Evaluation(args)
    # ROC Curve(None)
    evalator.roc_auc(networks, data, 'None')
    # ROC Curve(-15)
    evalator.roc_auc(networks, data, '-15')
    # ROC Curve(-3)
    evalator.roc_auc(networks, data, '-3')
    #
    # # Fixed FRR
    # evalator.far(networks, data, frr=1)
    #
    # # Qualitative results
    # evalator.netvad(net, data, only_plot_net=False, net_name='RNN')
    # print('Complete RNN')
    # evalator.netvad(net_large, data, only_plot_net=False, net_name='RNN(large)')
    # print('Complete RNN (Large)')
    # evalator.netvad(net_large, data, only_plot_net=False, net_name='Conv + RNN')
    # print('Complete Conv + RNN')
    # evalator.netvad(net_large, data, only_plot_net=False, net_name='Conv + RNN (large)')
    # print('Complete Conv + RNN (large)')
    # evalator.netvad(net_large, data, only_plot_net=False, net_name='DenseNet')
    # print('Complete DenseNet')
    # evalator.netvad(net_large, data, only_plot_net=False, net_name='DenseNet (large)')
    # print('Complete DenseNet (large)')
