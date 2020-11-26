import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from loss import FocalLoss
from utils import DataGenerator

obj_cuda = torch.cuda.is_available()


def accuracy(out, y):
    """
    Calculate accuracy of model where
    out.shape = (64, 2) and y.shape = (64)
    """
    out = torch.max(out, 1)[1].float()
    eq = torch.eq(out, y.float()).float()
    return torch.mean(eq)


def net_path(epoch, title):
    part = os.getcwd() + '/models/' + title + '/' + title
    if epoch >= 0:
        return part + '_epoch' + str(epoch).zfill(3) + '.net'
    else:
        return part + '.net'


def save_net(net, epoch, title='net'):
    if not os.path.exists(os.getcwd() + '/models/'+ title):
        os.makedirs(os.getcwd() + '/models/' + title)
    torch.save(net, net_path(epoch, title))


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.noise_levels = ['None', '-15', '-3']
        self.levels = None
        self.batch_size = args.batch_size
        self.frame_count = args.frames
        self.step_size = args.step_size

    def train(self, net, data, size_limit=0, noise_level='None', epochs=15, lr=1e-3, use_adam=True,
              weight_decay=1e-5, momentum=0.9, use_focal_loss=True, gamma=0.0,
              early_stopping=False, patience=25, auto_save=True, title='net', verbose=True):
        # set up an instance of data generator using default partitions
        generator = DataGenerator(self.args, data, size_limit)
        generator.setup_generation(self.frame_count, self.step_size, self.batch_size)

        # Noise level does not match
        if noise_level not in self.noise_levels:
            raise Exception('Error: invalid noise level!')

        # When the training data connot be found
        if generator.train_size == 0:
            raise Exception('Error: no training data was found')

        # Instantiate the chosen loss function
        if use_focal_loss:
            criterion = FocalLoss(gamma)
            self.levels = self.noise_levels
        else:
            criterion = nn.CrossEntropyLoss()
            self.levels = [noise_level]

        # Move network, criterion to GPU if available
        if obj_cuda:
            net.cuda()
            criterion.cuda()

        # Instantiate the chosen optimizer with the parameters specified
        if use_adam:
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        # If verbose print starting conditions
        if verbose:
            print(f'Initiating training of {title}...\n\nLearning rate: {lr}')
            _trsz = generator.train_size * 3 if use_focal_loss else generator.train_size
            _vlsz = generator.val_size * 3 if use_focal_loss else generator.val_size
            print(f'Model parameters: {sum(p.numel() for p in net.parameters())}')
            print(f'Frame partitions: {_trsz} | {_vlsz}')
            _critstr = f'Focal Loss (γ = {gamma})' if use_focal_loss \
                else f'Cross-Entropy ({noise_level} dB)'
            _optmstr = f'Adam (decay = {weight_decay})' if use_adam \
                else f'SGD (momentum = {momentum})'
            _earlstr = f'Early Stopping (patience = {patience})' if early_stopping else str(epochs)
            _autostr = 'Enabled' if auto_save else 'DISABLED'
            print(f'Criterion: {_critstr}\nOptimizer: {_optmstr}')
            print(f'Max epochs: {_earlstr}\nAuto-save: {_autostr}')

        net.train()
        stalecount, maxacc = 0, 0

        losses, accs, val_losses, val_accs = [], [], [], []

        if verbose:
            start_time = time.time()

        # Iterate over training epochs
        for epoch in range(epochs):
            # Calculate loss and accuracy for that epoch and optimize
            generator.use_train_data()
            loss, acc = self.run(net, generator, criterion, optimize=True, optimizer=optimizer)
            losses.append(loss)
            accs.append(acc)

            # If validation data is available, calculate validation metrics
            if generator.val_size != 0:
                net.eval()
                generator.use_validate_data()
                val_loss, val_acc = self.run(net, generator, criterion)

                val_losses.append(val_loss)
                val_accs.append(val_acc)
                net.train()

                # Early stopping algorithm.
                # If validation accuracy does not improve for
                # a set amount of epochs, abort training and retrieve
                # the best model (according to validation accuracy)
                if epoch > 0 and val_accs[-1] <= maxacc:
                    stalecount += 1
                    if stalecount > patience and early_stopping:
                        return
                else:
                    stalecount = 0
                    maxacc = val_accs[-1]

            if auto_save:
                save_net(net, epoch, title)

            # Optionally plot performance metrics continously
            if verbose:
                # Print measured wall-time of first epoch
                if epoch == 0:
                    dur = str(int((time.time() - start_time) / 60))
                    print(f'\nEpoch wall-time: {dur} min')

                # self.plot(losses, accs, val_losses, val_accs, generator)

    def plot(self, losses, accs, val_losses, val_accs, generator):
        """
        Continously plots the training/validation loss and accuracy of the model being trained.
        This functions is only called if verbose is True for the training session.
        """
        e = [i for i in range(len(losses))]
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(e, losses, label='Loss (Training)')

        if generator.val_size != 0:
            plt.plot(e, val_losses, label='Loss (Validation)')

        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(e, accs, label='Accuracy (Training)')

        if generator.val_size != 0:
            plt.plot(e, val_accs, label='Accuracy (Validation)')

        plt.legend()
        plt.show()
        clear_output(wait=True)

    def run(self, net, generator, criterion, optimize=False, optimizer=None):
        """
        This function constitutes a single epoch.
        Snippets are loaded into memory and their associated
        frames are loaded as generators. As training progresses
        and new frames are needed, they are generated by the iterator,
        and are thus not stored in memory when not used.
        If optimize is True, the associated optimizer will backpropagate
        and adjust network weights.
        Returns the average sample loss and accuracy for that epoch.
        """
        epoch_loss, epoch_acc, level_acc = 0, [], []

        # In case we apply focal loss, we want to include all noise levels
        batches = generator.batch_count
        num_batches = batches * len(self.levels)

        if num_batches == 0:
            raise ValueError('Not enough data to create a full batch!')

        # Helper function responsible for running a batch
        def run_batch(X, y, epoch_loss, epoch_acc):

            X = Variable(torch.from_numpy(np.array(X)).float())
            y = Variable(torch.from_numpy(np.array(y))).long()

            if obj_cuda:
                X = X.cuda()
                y = y.cuda()

            out = net(X)

            # Compute loss and accuracy for batch
            batch_loss = criterion(out, y)
            batch_acc = accuracy(out, y)

            # If training session, initiate backpropagation and optimization
            if optimize:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if obj_cuda:
                batch_acc = batch_acc.cpu()
                batch_loss = batch_loss.cpu()

            # Accumulate loss and accuracy for epoch metrics
            epoch_loss += batch_loss.data.numpy() / float(self.batch_size)
            epoch_acc.append(batch_acc.data.numpy())

            return epoch_loss, epoch_acc

        # For each noise level scheduled
        for lvl in self.levels:

            # Set up generator for iteration
            generator.set_noise_level_db(lvl)

            # For each batch in noise level
            for i in range(batches):
                # Get a new batch and run it
                X, y = generator.get_batch(i)
                temp_loss, temp_acc = run_batch(X, y, epoch_loss, epoch_acc)
                epoch_loss += temp_loss / float(num_batches)
                level_acc.append(np.mean(temp_acc))

        return epoch_loss, np.mean(level_acc)
