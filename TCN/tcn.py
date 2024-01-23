import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Initialize convolutional kernels as an exp2syn with random decays
        # with torch.no_grad():
        #     self.conv1.weight.copy_(self.exp2syn_weights(self.conv1.weight.data))
        #     self.conv2.weight.copy_(self.exp2syn_weights(self.conv2.weight.data))
        #     if self.downsample is not None:
        #         self.downsample.weight.data.normal_(0, 0.01)

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def exp2syn_weights(self, weight):
        weight_shape = weight.shape

        t_vec = torch.arange(0, weight_shape[2] + 1, 1).tile((weight_shape[0], weight_shape[1], 1))
        tau1 = torch.rand((weight_shape[1], weight_shape[0])).tile((weight_shape[2] + 1, 1, 1)).transpose(0,2) * 20
        tau2 = tau1 + (torch.rand((weight_shape[1], weight_shape[0])).tile((weight_shape[2] + 1,1,1)).transpose(0,2) * 20)

        return self.get_kernel(t_vec, tau1, tau2).flip(dims=(2,))[:, :, :-1] * torch.rand((1,))


    def get_kernel(self, t_vec, tau1, tau2):
        G = tau2/(tau2-tau1)*(-torch.exp(-t_vec/tau1) + torch.exp(-t_vec/tau2))
        return G

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, seq_len):
        super(TCN, self).__init__()
        self.hidden_size = 128
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear1 = nn.Linear(num_channels[-1], self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, output_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.float()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.tanh(output)
        output = self.linear1(output)
        output = self.tanh(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        output = self.tanh(output)
        output = self.dropout2(output)
        output = self.linear3(output)
        return output