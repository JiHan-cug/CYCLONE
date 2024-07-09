import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        layer = nn.Linear(layers[i - 1], layers[i])
        nn.init.kaiming_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0)
        net.append(layer)

        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "mish":
            net.append(Mish())
        elif activation == "tanh":
            net.append(nn.Tanh())
    return nn.Sequential(*net)

class Con_l(nn.Module):
    def __init__(self, input_dim, domin,
                 device, z_dim=32, h_dim=16,
                 encode_layers=[512, 256],
                 decode_layers=[256, 512],
                 activation='relu'):
        super(Con_l, self).__init__()

        self.pretrain = True
        self.domin = domin
        self.droplayer = nn.Dropout(0.7)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.activation = activation
        self.device = device

        batch_dim = z_dim
        z_dim_2 = z_dim + batch_dim

        self.encoder_n = buildNetwork([input_dim + batch_dim] + encode_layers, activation=activation)
        self.encoder_won = buildNetwork([input_dim] + encode_layers, activation=activation)
        self.decoder = buildNetwork([z_dim_2] + decode_layers, activation=activation)

        self.enc_b = nn.Linear(domin, batch_dim, bias=False)

        self.enc_mu = nn.Linear(encode_layers[-1], z_dim)
        self.enc_var = nn.Linear(encode_layers[-1], z_dim)
        self.enc_z = nn.Linear(encode_layers[-1], z_dim)

        self.enc_con = nn.Linear(encode_layers[-1], z_dim)

        self.dec_x = nn.Linear(decode_layers[-1], input_dim)

        self.projector = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )

    def Encoder(self, x, x1, x2, b):

        if x1 is not None and x2 is not None:
            h = self.encoder_won(x)
            z = self.enc_z(h)
            q_h = self.encoder_won(x1)
            k_h = self.encoder_won(x2)
            q_z = self.enc_con(q_h)
            k_z = self.enc_con(k_h)

            return z, q_z, k_z

        else:
            h = self.encoder_won(x)
            z = self.enc_z(h)
            return z

    def Decoder(self, z, b):

        b_enc = self.enc_b(b)
        z_b = torch.cat([z, b_enc], dim=1)

        h = self.decoder(z_b)
        x_x = self.dec_x(h)

        return x_x

    def forward(self, x, b, x1, x2, y):

        z = self.Encoder(x=x, x1=None, x2=None, b=b)

        if self.pretrain:
            q_z = self.Encoder(x=x, x1=x1, x2=None, b=None)
            k_z = self.Encoder(x=x, x1=None, x2=x2, b=None)
            q_z1 = self.projector(q_z)
            k_z1 = self.projector(k_z)

            x_x = self.Decoder(z, b)

            return z, q_z1, k_z1, x_x


    def EncodeAll(self, X, b, batch_size=256):
        all_z_mu = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            exp = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            exp = torch.tensor(np.float32(exp))
            b_1 = b[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            b_1 = torch.tensor(np.float32(b_1))
            with torch.no_grad():
                z_mu = self.Encoder(x=exp.to(self.device), x1=None, x2=None, b=b_1.to(self.device))

            all_z_mu.append(z_mu)

        all_z_mu = torch.cat(all_z_mu, dim=0)
        return all_z_mu

