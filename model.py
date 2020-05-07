import torch
import torch.nn as nn


class ApproxEMD(nn.Module):
    def __init__(self, n_in, n_hidden):
        self.n_hidden = n_hidden
        self.emb_layer = MLP(input_hidden=n_in, n_hiddens=[n_hidden, n_hidden])
        self.out_layer = MLP(input_hidden=n_hidden * 3, n_hiddens=[n_hidden, n_hidden])
        self.final_layer = nn.Linear(n_hidden, 1)
        self.final_act = nn.ReLU()
        return

    def forward(self, x, y):
        h_1 = self.emb_layer(x)  # [batch, n_h]
        h_2 = self.emb_layer(y)  # [batch, n_h]
        h = torch.stack([h_1, h_2, h_1 * h_2], dim=-1)  # [batch, 3*n_h]
        out = self.out_layer(h)  # [batch, 1]

        approx_dist = self.final_act(self.final_layer(out))
        return approx_dist.reshape(-1)  # [batch]


class MLP(nn.Module):
    def __init__(self, input_hidden, n_hiddens=[64, 64], fix_output_dim=True):
        super().__init__()
        tmp_hiddens = n_hiddens.copy()
        if fix_output_dim:
            tmp_hiddens.append(input_hidden // 2)

        self.n_hiddens = tmp_hiddens
        self.layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        h_in = input_hidden
        for i, h_out in enumerate(tmp_hiddens):
            self.layers.append(nn.Linear(h_in, h_out))
            if i < len(tmp_hiddens) - 1:
                self.act_layers.append(nn.ReLU())
            h_in = h_out
        return

    def forward(self, x):
        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](h)
            if i < len(self.layers) - 1:
                h = self.act_layers[i](h)
        return h
