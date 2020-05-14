import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
from torch.autograd import Variable


class ApproxEMD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "SymAtt_Double"
        self.ratio = args.ratio
        self.n_hidden = n_hidden = args.n_hidden

        self.emb_layer = MLP(input_hidden=300, n_hiddens=[n_hidden])

        self.query_layer = MLP(input_hidden=n_hidden * 3, n_hiddens=[n_hidden, n_hidden])
        self.query_act = nn.ReLU()

        self.att_read = BatchScaledDotProductAttention(n_in=n_hidden, n_out=n_hidden)
        self.att_comp = BatchScaledDotProductAttention(n_in=n_hidden, n_out=n_hidden)

        self.mid_layer = MLP(input_hidden=n_hidden, n_hiddens=[n_hidden, n_hidden], final_act=True)

        self.out_layer = MLP(input_hidden=n_hidden * 3, n_hiddens=[n_hidden, n_hidden], final_act=True)
        self.out_act = nn.ReLU()
        self.final_layer = nn.Linear(n_hidden, 1)
        self.ratio_act = nn.Sigmoid()

    def forward(self, sentences_1, sentences_2):
        seq_1, seq_len_1 = pad_packed_sequence(sentences_1, batch_first=True)
        seq_2, seq_len_2 = pad_packed_sequence(sentences_2, batch_first=True)
        if torch.cuda.is_available():
            div_seq_len_1 = seq_len_1.cuda(); div_seq_len_2 = seq_len_2.cuda()

        # [batch, length, n_dim]
        seq_1 = self.emb_layer(seq_1)
        seq_2 = self.emb_layer(seq_2)

        q_1 = torch.cat([torch.max(seq_1, dim=1)[0], torch.min(seq_1, dim=1)[0], torch.sum(seq_1, dim=1) / div_seq_len_1.view(-1, 1)], dim=-1)
        q_1 = self.query_act(self.query_layer(q_1))  # [batch, n_hidden]
        q_2 = torch.cat([torch.max(seq_2, dim=1)[0], torch.min(seq_2, dim=1)[0], torch.sum(seq_2, dim=1) / div_seq_len_2.view(-1, 1)], dim=-1)
        q_2 = self.query_act(self.query_layer(q_2))  # [batch, n_hidden]

        v_1 = self.att_read(q_2, seq_1, seq_len_1)
        v_2 = self.att_read(q_1, seq_2, seq_len_2)

        seq_1 = self.mid_layer(seq_1)
        seq_2 = self.mid_layer(seq_2)

        h_2 = self.att_comp(v_1, seq_2, seq_len_2)
        h_1 = self.att_comp(v_2, seq_1, seq_len_1)

        h_ab = torch.cat([h_1, h_2, h_1 - h_2], dim=-1)  # [bs, 3 * n_h]
        h_ba = torch.cat([h_2, h_1, h_2 - h_1], dim=-1)  # [bs, 3 * n_h]
        h_final = self.out_layer(h_ab) + self.out_layer(h_ba) / 2.
        d = self.final_layer(h_final).view(-1)  # [batch]

        if self.ratio:
            d = self.ratio_act(d)

        return d


class ApproxEMDAttentionDouble(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "AttGRU_Double"
        self.ratio = args.ratio
        self.n_hidden = n_hidden = args.n_hidden
        self.gru_read = nn.GRU(input_size=300, hidden_size=n_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.gru_comp = nn.GRU(input_size=n_hidden * 2, hidden_size=n_hidden, num_layers=1, bidirectional=True, batch_first=True)

        self.att_read = BatchScaledDotProductAttention(n_in=n_hidden*2, n_out=n_hidden*2)
        self.att_comp = BatchScaledDotProductAttention(n_in=n_hidden*2, n_out=n_hidden)

        self.out_layer = MLP(input_hidden=n_hidden*3, n_hiddens=[n_hidden, n_hidden])
        self.out_act = nn.ReLU()
        self.final_layer = nn.Linear(n_hidden, 1)
        self.ratio_act = nn.Sigmoid()

    def forward(self, sentences_1, sentences_2):
        seq_1, hidden_1 = self.gru_read(sentences_1)
        hidden_1 = torch.transpose(hidden_1, 0, 1)  # [batch, 2, n_h]
        bs = hidden_1.size()[0]
        hidden_1 = hidden_1.reshape(bs, -1)
        output_1, seq_len_1 = pad_packed_sequence(seq_1, batch_first=True)

        seq_2, hidden_2 = self.gru_read(sentences_2)
        hidden_2 = torch.transpose(hidden_2, 0, 1).reshape(bs, -1)  # [batch, 2, n_h]
        output_2, seq_len_2 = pad_packed_sequence(seq_2, batch_first=True)

        att_read_1 = self.att_read(hidden_2, output_1, seq_len_1)
        att_read_2 = self.att_read(hidden_1, output_2, seq_len_2)

        seq_1, hidden_1 = self.gru_comp(seq_1)
        seq_2, hidden_2 = self.gru_comp(seq_2)

        seq_1, seq_len_2 = pad_packed_sequence(seq_1, batch_first=True)
        seq_2, seq_len_2 = pad_packed_sequence(seq_2, batch_first=True)

        att_1 = self.att_comp(att_read_2, seq_1, seq_len_1)
        att_2 = self.att_comp(att_read_1, seq_2, seq_len_2)

        h = torch.cat([att_1, att_2, att_1 - att_2], dim=-1)  # [bs, 6 * n_h]
        h_final = self.out_act(self.out_layer(h))  # [batch, 1]

        d = self.final_layer(h_final).view(-1)  # [batch]

        if self.ratio:
            d = self.ratio_act(d)

        return d


class ApproxEMDAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "AttGRU"
        self.ratio = args.ratio
        self.n_hidden = n_hidden = args.n_hidden
        self.gru = nn.GRU(input_size=300, hidden_size=n_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.att = BatchScaledDotProductAttention(n_in=n_hidden*2, n_out=n_hidden)
        self.out_layer = MLP(input_hidden=n_hidden * 3, n_hiddens=[n_hidden, n_hidden])
        self.out_act = nn.ReLU()
        self.final_layer = nn.Linear(n_hidden, 1)
        self.ratio_act = nn.Sigmoid()

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

        d = self.final_layer(h_final).view(-1)  # [batch]

        if self.ratio:
            d = self.ratio_act(d)

        return d


class ApproxEMDBaseline(nn.Module):
    def __init__(self, args):
        self.name = "Baseline"
        super().__init__()
        self.ratio = args.ratio
        self.n_hidden = n_hidden = args.n_hidden
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

        d = self.final_layer(h_final).view(-1)  # [batch]
        if self.ratio:
            d = self.ratio_act(d)
        return d


# Batch attention
class BatchScaledDotProductAttention(nn.Module):
    def __init__(self, n_in, n_out):
        '''scaled-dot-product-attention
            parameters:
                d_model: A scalar. attention size
        '''
        super(BatchScaledDotProductAttention, self).__init__()
        self.temper = np.power(n_in, 0.5)
        self.q_layer = nn.Linear(n_in, n_out)
        self.k_layer = nn.Linear(n_in, n_out)
        self.v_layer = nn.Linear(n_in, n_out)
        self.out_linear = nn.Linear(n_out, n_out)

    def forward(self, h, seq, seq_length):
        ''' forward step
            parameters:
                h ( ... * d)
                seq ( ... * len_2 * d)
                seq_length ( ... )
            note: dv == dk
        '''
        # print("Attn_1")
        h = self.q_layer(h).unsqueeze(-1)  # [ batch * h * 1 ]
        k = self.k_layer(seq)  # [ batch, length, h ]
        v = self.v_layer(seq)  # [ batch, length, h ]

        # print("Attn_2")
        scores = torch.matmul(k, h)  # [batch, length, 1]
        # print("scores", scores.size())
        scores = scores.squeeze(2)  # [ batch, length ]
        # print("scores", scores.size())

        # print("Attn_3")
        kv_mask = self.get_a_mask(seq_length)  # [ batch, length ]
        # print("kv_mask", kv_mask.size())

        # print("Attn_4")
        scores = scores.masked_fill(kv_mask == 0, -1e9)

        # print("Attn_5")
        weight = F.softmax(scores / self.temper, dim=-1).unsqueeze(1)
        # print("weight", weight.size())  # [batch, 1, length]
        output = torch.matmul(weight, v)  # [batch, 1, length] -> [batch, 1, h]
        # print("output", output.size())  # [batch, 1, length]

        # print("Attn_6")
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
    def __init__(self, input_hidden, n_hiddens=[64, 64], final_act=False):
        super().__init__()
        self.n_hiddens = n_hiddens.copy()
        self.final_act = final_act
        self.layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        h_in = input_hidden
        for i, h_out in enumerate(self.n_hiddens):
            self.layers.append(nn.Linear(h_in, h_out))
            if i < len(self.n_hiddens) - 1:
                self.act_layers.append(nn.ReLU())
            h_in = h_out
        if final_act:
            self.act_layers.append(nn.ReLU())
        return

    def forward(self, x):
        h = x
        for i in range(len(self.layers)):
            # print("MLP, i", i, " h", h.size())
            h = self.layers[i](h)
            if i < len(self.layers) - 1:
                h = self.act_layers[i](h)
        if self.final_act:
            h = self.act_layers[-1](h)
        return h
