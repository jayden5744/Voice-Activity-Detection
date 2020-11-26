import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, RNN, LSTM, GRU

obj_cuda = torch.cuda.is_available()


def num_params(net, verbose=True):
    count = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f'Model parameters: {count}')
    return count


class Net(nn.Module):
    def __init__(self, args, large=True, lstm=True):
        super(Net, self).__init__()
        self.args = args
        self.large = large
        self.lstm = lstm
        self.relu = nn.ReLU()
        self.frames = args.frames
        self.batch_size = args.batch_size

        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = LSTM(input_size=args.features, hidden_size=args.frames, num_layers=1, batch_first=True)
        else:
            self.rnn = GRU(input_size=args.features, hidden_size=args.frames, num_layers=1, batch_first=True)

        if large:
            self.lin1 = nn.Linear(args.frames ** 2, 26)
            self.lin2 = nn.Linear(26, 2)
        else:
            self.lin = nn.Linear(args.frames ** 2, 2)

        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        h = Variable(torch.zeros(1, self.batch_size, self.frames))
        c = Variable(torch.zeros(1, self.batch_size, self.frames))

        if obj_cuda:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, x):
        # if obj_cuda:
        #    self.rnn.flatten_parameters()

        # (batch, frames, features)
        if hasattr(self, 'lstm') and self.lstm:
            x, _ = self.rnn(x, self.hidden)
        else:
            x, _ = self.rnn(x)

        x = x.contiguous().view(-1, self.frames ** 2)

        # (batch, units)
        if self.large:
            x = self.relu(self.lin1(x))
            x = self.lin2(x)
        else:
            x = self.lin(x)

        return self.softmax(x)


if __name__ == '__main__':
    net = Net(large=False)
    num_params(net)
    print(net)