import torch
import torch.nn as nn



class ApproxEMD(nn.Module):

    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.gru = nn.RNN(input_size=300, hidden_size=n_hidden, num_layers=1, bidirectional=False, batch_first=True)

        self.out_layer = MLP(input_hidden=n_hidden * 3, n_hiddens=[n_hidden, n_hidden])
        self.final_layer = nn.Linear(n_hidden, 1)
        self.final_act = nn.ReLU()
        return

    def forward(self, sentences_1, sentences_2):
        bs = sentences_1.size()[0]
        # print("bs", bs)
        # print("sentences_1", sentences_1.shape)

        _, hidden_1 = self.gru(sentences_1)
        hidden_1 = hidden_1.view(bs, -1)

        _, hidden_2 = self.gru(sentences_2)
        hidden_2 = hidden_2.view(bs, -1)

        h = torch.cat([hidden_1, hidden_2, hidden_1 * hidden_2], dim=-1)  # [bs, 3 * n_h]
        h_final = self.out_layer(h)  # [batch, 1]

        approx_dist = self.final_act(self.final_layer(h_final))

        return approx_dist.view(-1)  # [batch]


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
