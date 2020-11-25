import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, RNN, LSTM, GRU, Conv1d, Conv2d, Dropout, BatchNorm1d, BatchNorm2d

obj_cuda = torch.cuda.is_available()


def num_params(net, verbose=True):
    count = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f'Model parameters: {count}')
    return count


class BiRNN(nn.Module):
    """
    Bi-directional layer of gated recurrent units.
    Includes a fully connected layer to produce binary output.
    """

    def __init__(self, num_in, num_hidden, batch_size=2048, large=True, lstm=False, fcl=True, bidir=False):
        super(BiRNN, self).__init__()

        self.num_hidden, self.batch_size = num_hidden, batch_size
        self.lstm, self.bidir, self.layers = lstm, bidir, 2 if large else 1

        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = LSTM(num_in, num_hidden, num_layers=self.layers, bidirectional=self.bidir, batch_first=True)
            sz = 18 if large else 16
        else:
            self.rnn = GRU(num_in, num_hidden, num_layers=self.layers, bidirectional=self.bidir, batch_first=True)
            sz = 18

        embed_sz = num_hidden * 2 if self.bidir or self.layers > 1 else num_hidden

        if not fcl:
            self.embed = nn.Linear(embed_sz, 2)
        else:
            if large:
                self.embed = nn.Sequential(
                    nn.Linear(embed_sz, sz + 14),
                    nn.BatchNorm1d(sz + 14),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz + 14, sz),
                    nn.BatchNorm1d(sz),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz, 2)
                )
            else:
                self.embed = nn.Sequential(
                    nn.Linear(embed_sz, sz),
                    nn.BatchNorm1d(sz),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz, 2)
                )

    def init_hidden(self):
        num_dir = 2 if self.bidir or self.layers > 1 else 1
        h = Variable(torch.zeros(num_dir, self.batch_size, self.num_hidden))
        c = Variable(torch.zeros(num_dir, self.batch_size, self.num_hidden))

        if obj_cuda:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, x):
        if obj_cuda:
            self.rnn.flatten_parameters()

        x = x.permute(0, 2, 1)

        if self.lstm:
            x, self.hidden = self.rnn(x, self.hidden)
        else:
            x, self.hidden = self.rnn(x)

        # Extract outputs from forward and backward sequence and concatenate
        # If not bidirectional, only use last output from forward sequence
        x = self.hidden.view(self.batch_size, -1)

        # (batch, features)
        return self.embed(x)


class GatedConv(nn.Module):
    """
    Gated convolutional layer using tanh as activation and a sigmoidal gate.
    The convolution is padded to keep its original dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=True):
        super(GatedConv, self).__init__()

        padding = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )
        self.conv_gate = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x) * self.conv_gate(x)


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=True):
        super(Conv, self).__init__()

        padding = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class GatedResidualConv(nn.Module):
    """
    Legacy class.
    Gated residual convolutional layer using tanh as activation and a sigmoidal gate.
    Outputs the accumulated input to be used in the following layer, as well as a
    residual connection that is added to the output of the following layer using
    element-wise multiplication. Input and output sizes are unchanged.
    """

    def __init__(self, channels, kernel_size=3, dilation=1):
        super(GatedResidualConv, self).__init__()

        self.gated_conv = GatedConv(channels, channels)

    def forward(self, x, r=None):
        # Residual connection defaults to x
        if r is None:
            r = x
        out = self.gated_conv(x)

        # (acummulated input, residual connection)
        return out * x, out * r


class NickNet(nn.Module):
    """
    This network consists of (gated) convolutional layers,
    followed by a bi-directional recurrent layer and one or
    more fully connected layers. Output is run through a
    softmax function.
    """

    def __init__(self, args, large=True, residual_connections=False, gated=True, lstm=False,
                 fcl=True, bidir=False):
        super(NickNet, self).__init__()

        frames = args.framse
        features = args.features
        batch_size = args.batch_size

        self.large = large
        self.residual_connections = residual_connections

        # Define number of channels depending on configuration.
        # This is done to ensure that number of parameters are
        # held some-what constant for all model configurations.
        if large:
            if gated:
                conv_channels1, conv_channels2, conv_channels3, conv_channels4 = 32, 28, 25, 18
            else:
                conv_channels1, conv_channels2, conv_channels3, conv_channels4 = 38, 35, 31, 24
            conv_channels_out = conv_channels4
        else:
            if gated:
                conv_channels1, conv_channels2, conv_channels3 = 20, 18, 16
            else:
                conv_channels1, conv_channels2, conv_channels3 = 26, 20, 16
            conv_channels_out = conv_channels3

        # Gated convolution with residual connections
        if residual_connections:
            conv_channels3 = conv_channels2
            self.conv1 = GatedConv(features, conv_channels3)
            self.conv2 = GatedResidualConv(conv_channels3)
            self.conv3 = GatedResidualConv(conv_channels3)
            if large:
                self.conv4 = GatedResidualConv(conv_channels3)

        # Gated convolution
        elif gated:
            self.conv1 = GatedConv(features, conv_channels1)
            self.conv2 = GatedConv(conv_channels1, conv_channels2)
            self.conv3 = GatedConv(conv_channels2, conv_channels3)
            if large:
                self.conv4 = GatedConv(conv_channels3, conv_channels4)

        # Default convolution
        else:
            self.conv1 = Conv(features, conv_channels1)
            self.conv2 = Conv(conv_channels1, conv_channels2)
            self.conv3 = Conv(conv_channels2, conv_channels3)
            if large:
                self.conv4 = Conv(conv_channels3, conv_channels4)

        # Recurrent layer
        num_hidden = conv_channels_out + 11 if large else conv_channels_out + 5
        self.rnn = BiRNN(conv_channels_out, num_hidden, batch_size=batch_size,
                         large=large, lstm=lstm, fcl=fcl, bidir=bidir)

    def forward(self, x):
        # (batch, frames, features)
        x = x.permute(0, 2, 1)

        # (batch, features/channels, frames)
        x = self.conv1(x)

        if self.residual_connections:
            x, r = self.conv2(x)
            x, r = self.conv3(x, r)
            if self.large:
                x, r = self.conv4(x, r)
            x = x * r
        else:
            x = self.conv2(x)
            x = self.conv3(x)
            if self.large:
                x = self.conv4(x)

        #   (batch, channels, frames)
        # ->(batch, frames, channels)
        x = self.rnn(x)

        # (batch, 2)
        return F.softmax(x, dim=1)

if __name__ == '__main__':
    gru = NickNet(large=True)
    num_params(gru)
    print(gru)