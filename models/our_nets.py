import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical


class SingleTaskNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_window=1, mask_mode=None, encoder_archs=None, decoder_arch=None, model_tag=None):
        super().__init__()
        # data dimension
        self.model_tag = model_tag
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.encoder = None
        self.gmm_network = None
        self.clf = None

        # mask
        self.mask_mode = mask_mode
        self.mask = None
        

        if mask_mode is not None:
            self.mask = torch.ones((1,(dim_window+1)*dim_in))
            self.mask[0, (dim_window*dim_in)+mask_mode:] = 0
            print('mask', self.mask)
        else:
            self.mask = torch.ones((1,(dim_window+1)*dim_in))
            self.mask[0, (dim_window*dim_in):] = 0

        # curr_in = dim_in * dim_window if mask_mode is None else dim_in * (dim_window+1)
        curr_in = dim_in * (dim_window+1)
        if encoder_archs is not None:
            pass
            
        assert decoder_arch[0] in ['gmm', 'softmax'], "Unknown Decoder Type"
        self.decoder_type = decoder_arch[0]
        if self.decoder_type == 'gmm':
            self.gmm_network = MixtureDensityNetwork(curr_in, dim_out, n_components=decoder_arch[1])
        else:
            pass

    def forward(self, x):
        # print('forward x', x.shape)
        # print('example x', x[0])
        out = self.make_mask(x)
        # print(out)
        # print('forward out', out.shape)
        # print('example x', out[0])
        if self.encoder is not None:        
            pass
        if self.gmm_network is not None:
            pi, normal = self.gmm_network(out)
            return pi, normal
    
    def loss(self, x, y):
        if self.decoder_type == 'gmm':
            pi, normal = self.forward(x)
            return self.gmm_network.loss(pi, normal, y)
        
    def sample(self, x):
        if self.decoder_type == 'gmm':
            pi, normal = self.forward(x)
            return self.gmm_network.sample(pi, normal)

    def make_mask(self, x):
        if self.mask_mode is None:
            # print('masking', x, self.dim_in)
            # x = x[:, :-self.dim_in]
            x = x * self.mask
        else:
            x = x * self.mask
        return x
        

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 1
        print(cfg)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)



class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = MixtureDiagNormalNetwork(dim_in, dim_out,
                                                       n_components)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, pi, normal, y):
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, pi, normal):
        # pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples, normal


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()
        self.n_components = n_components
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim * n_components),
        )

    def forward(self, x):
        params = self.network(x)
        # print(params)
        # input()
        mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)
