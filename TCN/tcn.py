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
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

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
        out = self.network(x)
        return out

class model_TCN(nn.Module):
    def __init__(self, input_size, output_size, seq_len, num_channels=[32]*3, kernel_size=20,
                 hidden_size=128, dropout=0.2, n_lstm_layers=2, lstm_hidden_dim=32):
        super(model_TCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim 

        self.hidden_size = hidden_size
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.tanh = nn.Tanh()

        layer_size = [hidden_size] * 3

        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.float()

        self.fc_mid = model_ann(num_channels[-1], lstm_hidden_dim, layer_size)
        self.lstm = nn.LSTM(input_size=num_channels[-1], hidden_size=lstm_hidden_dim, num_layers=n_lstm_layers, batch_first=True, dropout=dropout)
        self.fc_out = model_ann(lstm_hidden_dim, output_size, layer_size)

    def forward(self, x, h, c):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        out = self.fc_mid(out)
        out, (h, c) =  self.lstm(out, (h, c))
        out = self.fc_out(out)
        return out, h, c


class model_ann(nn.Module):
    def __init__(self, input_size, output_size, layer_size):
        super(model_ann, self).__init__()
        self.input_size,  self.layer_size, self.output_size = input_size, layer_size, output_size

        #List layer sizes
        self.layer_hidden = [input_size] + layer_size + [output_size]
        
        #Compile layers into lists
        layer_list = list()
        for idx in range(len(self.layer_hidden)-2):
            layer_list.append(nn.Linear(in_features=self.layer_hidden[idx], out_features=self.layer_hidden[idx+1]))
            layer_list.append(nn.Tanh())

        layer_list.append(nn.Linear(in_features=self.layer_hidden[-2], out_features=self.layer_hidden[-1]))
        
        self.fc = nn.Sequential(*layer_list)
        

    def forward(self, x):
        # #Encoding step
        # for idx in range(len(self.layer_list)):
        #     x = F.tanh(self.layer_list[idx](x))

        return self.fc(x)