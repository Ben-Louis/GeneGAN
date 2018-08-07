import torch
from torch.backends import cudnn
import os
import argparse
from models import *
from datasets import *
from utils import makedir
from train import train

def str2bool(s):
    return s.lower() == 'true'

# get parameters
def get_parameter():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--crop_size', type=int, default=158)
    parser.add_argument('--select_attrs', type=str, default='Bangs')

    # phase
    parser.add_argument('--phase', type=str, default='train')

    # model
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=str, default='[256,16]')

    # train
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=8)
    parser.add_argument('--d_train_repeat', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pretrained_model', type=int, default=-1)

    parser.add_argument('--lambda_rec', type=float, default=5)
    parser.add_argument('--lambda_zero', type=float, default=10)
    parser.add_argument('--lambda_paral', type=float, default=1)

    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--num_sample', type=int, default=10)
    parser.add_argument('--print_log', type=str2bool, default=True)

    parser.add_argument('--root', type=str, default='expr')
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--model_path', type=str, default='model')

    config = parser.parse_args()
    config.device = torch.device('cuda:0')
    config.select_attrs = config.select_attrs.split(',')
    config.latent_dim = eval(config.latent_dim)

    try:
        config.log_path = os.path.join(config.root, config.log_path)
    except FileExistsError:
        pass
    config.model_path = os.path.join(config.root, config.model_path)
    makedir(config.log_path)
    makedir(config.model_path)

    return config

def main():
    config = get_parameter()
    cudnn.benchmark = True

    ##### build model #####
    model = {}
    model['E'] = Encoder(config.conv_dim, config.latent_dim)
    model['G'] = Generator(config.conv_dim, config.latent_dim)
    model['D_pos'] = Discriminator(config.conv_dim, config.image_size)
    model['D_neg'] = Discriminator(config.conv_dim, config.image_size)

    ##### create dataset #####
    data = CelebaData_GeneGAN(config.data_root, config.image_size, config.crop_size, 
        config.select_attrs, config.phase.split('_')[0])

    ##### train/test #####
    train(model, data, config)


if __name__ == '__main__':
    main()