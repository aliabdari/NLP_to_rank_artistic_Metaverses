import torch
import torch.nn as nn


# class OneDimensionalCNN(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
#         super(OneDimensionalCNN, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool1d(kernel_size=2)
#         self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


class FCNet(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.15)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, feature_size)

    def forward(self, out, skip=False):
        out = out.to(torch.float32)
        if not skip:
            out = out.view(out.size(0), -1)
        out = self.relu1(self.fc1(out))
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        return out


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _, h_n = self.gru(x)
        if self.is_bidirectional:
            return h_n.mean(0)
        return h_n.squeeze(0)


# # per Videos
# class Autoenc(torch.nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(input_size, hidden_size),
#             torch.nn.Tanh()
#         )
#
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size, input_size),
#             torch.nn.Tanh()
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded, encoded


# class SelfAttentionModel(nn.Module):
#     def __init__(self, dim=768, out_dim=256, n_heads=2, batch_first=True):
#         super().__init__()
#         self.fc_q = nn.Linear(dim, out_dim, bias=False)
#         self.fc_k = nn.Linear(dim, out_dim, bias=False)
#         self.fc_v = nn.Linear(dim, out_dim, bias=False)
#         self.sa = nn.MultiheadAttention(out_dim, num_heads=n_heads, batch_first=batch_first)
#         self.lin = FCNet(out_dim, out_dim)
#
#     def forward(self, x, mask_param=None):
#         if mask_param is not None:
#             attn_output, attn_output_weights = self.sa(self.fc_q(x),
#                                                        self.fc_k(x),
#                                                        self.fc_v(x),
#                                                        mask_param)
#             weighted_input = attn_output.sum(1)
#             return self.lin(weighted_input)
#         else:
#             attn_output, attn_output_weights = self.sa(self.fc_q(x),
#                                                        self.fc_k(x),
#                                                        self.fc_v(x))
#             return self.lin(attn_output)


class ConverterNN(nn.Module):
    def __init__(self, dim=512, out_dim=200):
        super(ConverterNN, self).__init__()
        self.fc = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, bidirectional=False):
        super(LSTMNetwork, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, output_size, num_layers, batch_first=True, bidirectional=bidirectional)

        # Define the fully connected layer
        # self.fc = nn.Linear(hidden_size, output_size)
        self.bidirectional = bidirectional

    def forward(self, packed_input):
        packed_input = packed_input.to(torch.float32)
        # Forward pass through the LSTM layer
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        if not self.bidirectional:
            output = h_n.squeeze(0)
        else:
            output = h_n.mean(0)

        return output
