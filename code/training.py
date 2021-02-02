# This is the script in charge of training the neural network. Initializes data loaders for the
# training and testing sets and runs a given number of epochs with either default or given
# configuration parameters.

import sys
import os
import warnings
import numpy as np
import argparse
import json
import cv2
import time
import torch
import torch.nn as nn
import dataset

from torch.autograd import Variable
from torchvision import datasets, transforms
from model import UNet, test_UNet_grayscale, test_UNet_color, test_UNet_color_with_IR
from library import save_checkpoint, calculate_matrix_mse

# -------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
# network model parameters
parser.add_argument('--learning_rate', action = 'store', type = float, default = 0.02)
parser.add_argument('--batch_size', action = 'store', type = int, default = 1)
parser.add_argument('--momentum', action = 'store', type = float, default = 0.90)
parser.add_argument('--nesterov', action = 'store', type = bool, default = False)
parser.add_argument('--weight_decay', action = 'store', type = float, default = 0.0)
parser.add_argument('--step_size', action = 'store', type = int, default = 10)
parser.add_argument('--gamma', action = 'store', type = float, default = 0.1)
parser.add_argument('--workers', action = 'store', type = int, default = 4)
# training utils
parser.add_argument('--seed', action = 'store', type = float, required = False, default = time.time())
parser.add_argument('--epoch_start', action = 'store', type = int, default = 0)
parser.add_argument('--epoch_end', action = 'store', type = int, default = 50)
parser.add_argument('--print_freq', action = 'store', type = int, default = 50)
parser.add_argument('--ir_enabled', action = 'store', type = bool, default = False)
parser.add_argument('--start_from_checkpoint', action = 'store', type = bool, default = False)
# filenames
parser.add_argument('--training_file', action = 'store', type = str, required = True)
parser.add_argument('--validating_file', action = 'store', type = str, required = True)
parser.add_argument('--checkpoint_filename', action = 'store', type = str, required = True)
parser.add_argument('--mse_history_filename', action = 'store', type = str, required = True)
# directories
parser.add_argument('--images_dir', action = 'store', type = str, required = True)
parser.add_argument('--labels_dir', action = 'store', type = str, required = True)

def main():
    
    global args
    args = parser.parse_args()
    best_mse = np.infty

    # print configuration
    print('configuration: {}.'.format(args))

    # files for training.
    with open(args.training_file, 'r') as outfile:        
        train_list = json.load(outfile)

    # files for testing.
    with open(args.validating_file, 'r') as outfile:
        val_list = json.load(outfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed(args.seed)
    
    # network model
    network = UNet(input_filters = 4) if args.ir_enabled else UNet()
    network = network.cuda()

    # initialize loss, optimized and learning rate scheduler    
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr = args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)

    # if start_from_checkpoint is enabled, continue training from checkpoint file
    # default: start from scratch
    if args.start_from_checkpoint:
        if os.path.isfile(args.checkpoint_filename):
            print("=> loading checkpoint '{}'".format(args.checkpoint_filename))
            checkpoint = torch.load(args.checkpoint_filename)
            args.epoch_start = checkpoint['epoch']
            best_mse = checkpoint['best_mse']
            network.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint_filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint_filename))

    for epoch in range(args.epoch_start, args.epoch_end):

        # run training and update learning rate
        train(train_list, network, loss, optimizer, epoch, lr_scheduler.get_lr()[-1], args.ir_enabled)
        lr_scheduler.step()
        
        # run testing
        current_mse = validate(val_list, network, args.ir_enabled)

        # save mse
        mse_history_file = open(args.mse_history_filename, 'a')
        mse_history_file.write(str(current_mse) + '\n') 
        mse_history_file.close()
        
        is_best = current_mse < best_mse
        best_mse = min(current_mse, best_mse)

        print('[{epoch:03d}] current_mse: {current_mse:.3f}, best_mse: {best_mse:.3f}.'
              .format(epoch = epoch, current_mse = current_mse, best_mse = best_mse))

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.checkpoint_filename,
            'state_dict': network.state_dict(),
            'best_mse': best_mse,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.checkpoint_filename)

def train(train_list, network, loss, optimizer, epoch, lr, ir_enabled):

    print('Training epoch {epoch:03d}.'.format(epoch = epoch))
    
    losses = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
            dataset.listDataset(
                    train_list,
                    transform = transforms.ToTensor(),
                    #transform = transforms.Compose([
                    #        transforms.ToTensor(), transforms.Normalize(
                    #                mean = [0.485, 0.456, 0.406],
                    #                std = [0.229, 0.224, 0.225]
                    #        ),
                    #]),
                    train = True,
                    seen = network.seen,
                    batch_size = args.batch_size,
                    num_workers = args.workers,
                    ir_enabled = ir_enabled,
                    images_dir = args.images_dir,
                    labels_dir = args.labels_dir
            ),
            batch_size = args.batch_size
    )

    print('Epoch {}, processed {} samples, lr {lr:.10f}.'.format(epoch, epoch * len(train_loader.dataset), lr = lr))
    
    # set network in training mode
    network.train()
    
    for i, (network_input, target)in enumerate(train_loader):
        
        # clear accumulated gradient
        optimizer.zero_grad()

        network_input = network_input.cuda()
        network_input = Variable(network_input)

        # run image through network
        output = network(network_input)
        
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        
        # calculate loss and update 
        loss_result = loss(output, target)
        losses.update(loss_result.item(), network_input.size(0))

        loss_result.backward()
        optimizer.step()    
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\tLoss {loss.val:.4f} (avg: {loss.avg:.4f}).'.format(epoch, i, len(train_loader), loss = losses))
    
def validate(val_list, network, ir_enabled):

    print ('Begin test')
    test_loader = torch.utils.data.DataLoader(
            dataset.listDataset(
                    val_list,
                    shuffle = False,
                    transform = transforms.ToTensor(),
                    #transform = transforms.Compose([
                    #        transforms.ToTensor(),transforms.Normalize(
                    #                mean = [0.485, 0.456, 0.406],
                    #                std = [0.229, 0.224, 0.225]),
                    #]),
                    ir_enabled = ir_enabled,
                    images_dir = args.images_dir,
                    labels_dir = args.labels_dir
            ),
            batch_size = args.batch_size
    )    
    
    # set network in evaluation mode
    network.eval()
    
    mse = 0
    
    # iterate through the testing dataset
    for _, (network_input, target) in enumerate(test_loader):

        network_input = network_input.cuda()
        network_input = Variable(network_input)
        
        output = network(network_input)
        output_matrix = output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3])

        # convert tensor object to matrix and reshape it
        output_matrix = np.asarray(output_matrix)
        target_matrix = np.asarray(target)
        target_matrix = target_matrix.reshape((output_matrix.shape[0], output_matrix.shape[1]))

        mse += calculate_matrix_mse(output_matrix, target_matrix)  

    return mse / len(test_loader) 

# Computes and stores the average and current values    
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()