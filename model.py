import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
from torch.autograd import Variable


class ApproxEMD(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.gru = nn.GRU(input_size=300, hidden_size=n_hidden, num_layers=1, bidirectional=True, batch_first=True)

        self.out_layer = MLP(input_hidden=n_hidden * 6, n_hiddens=[n_hidden, n_hidden])
        self.out_act = nn.ReLU()
        self.final_layer = nn.Linear(n_hidden, 1)
        self.final_act = nn.ReLU()
        return

    def forward(self, sentences_1, sentences_2):
        _, hidden_1 = self.gru(sentences_1)
        hidden_1 = torch.transpose(hidden_1, 0, 1)  # [batch, 2, n_h]
        bs = hidden_1.size()[0]
        # print(bs)
        hidden_1 = hidden_1.reshape(bs, -1)

        _, hidden_2 = self.gru(sentences_2)
        hidden_2 = torch.transpose(hidden_2, 0, 1)  # [batch, 2, n_h]
        hidden_2 = hidden_2.reshape(bs, -1)

        h = torch.cat([hidden_1, hidden_2, hidden_1 - hidden_2], dim=-1)  # [bs, 6 * n_h]
        h_final = self.out_act(self.out_layer(h))  # [batch, 1]

        return self.final_layer(h_final).view(-1)  # [batch]


class ApproxEMDAttention(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.gru = nn.GRU(input_size=300, hidden_size=n_hidden, num_layers=1, bidirectional=True, batch_first=True)

        self.att = BatchScaledDotProductAttention(n_hidden)

        self.out_layer = MLP(input_hidden=n_hidden * 3, n_hiddens=[n_hidden, n_hidden])
        self.out_act = nn.ReLU()
        self.final_layer = nn.Linear(n_hidden, 1)
        self.final_act = nn.ReLU()

    def forward(self, sentences_1, sentences_2):
        seq_1, hidden_1 = self.gru(sentences_1)
        hidden_1 = torch.transpose(hidden_1, 0, 1)  # [batch, 2, n_h]
        bs = hidden_1.size()[0]
        hidden_1 = hidden_1.reshape(bs, -1)
        output_1, output_lengths_1 = pad_packed_sequence(seq_1, batch_first=True)

        seq_2, hidden_2 = self.gru(sentences_2)
        hidden_2 = torch.transpose(hidden_2, 0, 1)  # [batch, 2, n_h]
        hidden_2 = hidden_2.reshape(bs, -1)
        output_2, output_lengths_2 = pad_packed_sequence(seq_2, batch_first=True)

        att_1 = self.att(hidden_2, output_1, output_lengths_1)
        att_2 = self.att(hidden_1, output_2, output_lengths_2)

        h = torch.cat([att_1, att_2, att_1 - att_2], dim=-1)  # [bs, 6 * n_h]
        h_final = self.out_act(self.out_layer(h))  # [batch, 1]

        return self.final_layer(h_final).view(-1)  # [batch]


# Batch attention
class BatchScaledDotProductAttention(nn.Module):
    def __init__(self, n_hidden):
        '''scaled-dot-product-attention
            parameters:
                d_model: A scalar. attention size
        '''
        super(BatchScaledDotProductAttention, self).__init__()
        self.temper = np.power(n_hidden, 0.5)
        self.q_layer = nn.Linear(n_hidden * 2, n_hidden)
        self.k_layer = nn.Linear(n_hidden * 2, n_hidden)
        self.v_layer = nn.Linear(n_hidden * 2, n_hidden)
        self.out_linear = nn.Linear(n_hidden, n_hidden)

    def forward(self, h, seq, seq_length):
        ''' forward step
            parameters:
                h ( ... * d)
                seq ( ... * len_2 * d)
                seq_length ( ... )
            note: dv == dk
        '''
        h = self.q_layer(h).unsqueeze(-1)  # [ batch * h * 1 ]
        k = self.k_layer(seq)  # [ batch, length, h ]
        v = self.v_layer(seq)  # [ batch, length, h ]
        # print("h", h.size())
        # print("k", k.size())
        scores = torch.matmul(k, h)  # [batch, length, 1]
        # print("scores", scores.size())
        scores = scores.squeeze(2)  # [ batch, length ]
        # print("scores", scores.size())

        kv_mask = self.get_a_mask(seq_length)  # [ batch, length ]

        scores = scores.masked_fill(kv_mask == 0, -1e9)

        weight = F.softmax(scores / self.temper, dim=-1).unsqueeze(1)
        # print("weight", weight.size())  # [batch, 1, length]
        output = torch.matmul(weight, v)  # [batch, 1, length] -> [batch, 1, h]
        output = self.out_linear(output.squeeze(1))  # [batch, h]
        output = F.elu(output)
        return output

    # Generate a boolean mask
    def get_a_mask(self, lengths):
        max_len = max(lengths)
        lengths = self.list_to_var(lengths)
        temp = torch.arange(max_len)
        if torch.cuda.is_available():
            temp = temp.cuda()
        mask = Variable((temp[None, :] < lengths[:, None]))
        return mask

    def list_to_var(self, x):
        x = Variable(torch.LongTensor(x))
        if torch.cuda.is_available():
            x = x.cuda()
        return x


class MLP(nn.Module):
    def __init__(self, input_hidden, n_hiddens=[64, 64]):
        super().__init__()
        self.n_hiddens = n_hiddens.copy()
        self.layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        h_in = input_hidden
        for i, h_out in enumerate(self.n_hiddens):
            self.layers.append(nn.Linear(h_in, h_out))
            if i < len(self.n_hiddens) - 1:
                self.act_layers.append(nn.ReLU())
            h_in = h_out
        return

    def forward(self, x):
        h = x
        for i in range(len(self.layers)):
            # print("MLP, i", i, " h", h.size())
            h = self.layers[i](h)
            if i < len(self.layers) - 1:
                h = self.act_layers[i](h)
        return h
