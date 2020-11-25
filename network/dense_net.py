import torch
import torch.nn as nn
import torch.nn.functional as F

obj_cuda = torch.cuda.is_available()


def num_params(net, verbose=True):
    count = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f'Model parameters: {count}')
    return count


class DenseSingle(nn.Module):

    def __init__(self, input_size, output_size, dropout, dilation, padding, kernel_size, stride):
        super(DenseSingle, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=padding,
                      stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):

    def __init__(self, input_size, n_layers, growth_rate, dropout, dilation, padding, kernel_size, stride):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(DenseSingle(input_size + i * growth_rate, growth_rate,
                                      dropout, dilation, padding, kernel_size, stride))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(TransitionBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(output_size),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Module):
    def __init__(self, large=False, batch_size=2048):
        super(DenseNet, self).__init__()
        self.batch_size = batch_size
        if large:

            dropout = 0.4

            self.cnn_in = nn.Sequential(
                nn.Conv1d(in_channels=24, out_channels=48, kernel_size=6, stride=1, padding=0,
                          dilation=4, bias=False),
                nn.BatchNorm1d(48),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.dense1 = DenseBlock(input_size=48, n_layers=8, growth_rate=4, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.trans1 = TransitionBlock(input_size=80, output_size=48, dropout=dropout)

            self.dense2 = DenseBlock(input_size=48, n_layers=8, growth_rate=4, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.cnn_out = nn.Sequential(
                nn.Conv1d(in_channels=80, out_channels=80, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(80),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.out = nn.Linear(80, 2, bias=False)

        else:
            dropout = 0.4

            self.cnn_in = nn.Sequential(
                nn.Conv1d(in_channels=24, out_channels=24, kernel_size=6, stride=1, padding=0,
                          dilation=4, bias=False),
                nn.BatchNorm1d(24),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.dense1 = DenseBlock(input_size=24, n_layers=6, growth_rate=3, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.trans1 = TransitionBlock(input_size=42, output_size=24, dropout=dropout)

            self.dense2 = DenseBlock(input_size=24, n_layers=6, growth_rate=3, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.cnn_out = nn.Sequential(
                nn.Conv1d(in_channels=42, out_channels=42, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(42),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.out = nn.Linear(42, 2, bias=False)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.cnn_in(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.cnn_out(x)

        x = x.view(self.batch_size, -1)

        return F.softmax(self.out(x), dim=1)


if __name__ == '__main__':
    densenet = DenseNet(large=True)
    num_params(densenet)
    print(densenet)