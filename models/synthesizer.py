import torch
import torch.optim as optim
import logging
import numpy as np

from models.our_nets import SingleTaskNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableFlowSynthesizer(object):
    def __init__(self, dim_in, dim_window, discrete_columns=[], learning_mode='A'):
        self.dim_in = dim_in
        self.dim_window = dim_window
        assert learning_mode in ['A', 'B'], "Unknown Mask Type"

        print('initing', self.dim_window, dim_in)
        self.models = []
        self.optimizers = []
        for col_i in range(dim_in):
            if col_i not in discrete_columns:
                mask_mode = None if learning_mode == 'A' else col_i 
                model_i = SingleTaskNet(2, 1, dim_window=self.dim_window, mask_mode=mask_mode, decoder_arch=['gmm', 2], model_tag=col_i)
                optim_i = optim.Adam(model_i.parameters(), lr=0.2) # 0.005
                self.models.append(model_i)
                self.optimizers.append(optim_i)


    def fit(self, X, y, epochs=100, batch=None):
        for col_i in range(self.dim_in):
            model = self.models[col_i]
            optimizer = self.optimizers[col_i]
            y_ = y[:, col_i].view(-1, 1)
            for i in range(epochs):
                optimizer.zero_grad()
                loss = model.loss(X, y_).mean()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    logger.info(f"Model: {col_i}\t" + f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    def sample(self, sample_num):
        p_samp = torch.zeros((1, (self.dim_window+1)*self.dim_in))

        gen_buff = []
        for i in range(sample_num):
            # print('input x_i', p_samp.size(), p_samp)
            for col_i in range(self.dim_in):
                model_i = self.models[col_i]
                sample_i, normal_i = model_i.sample(p_samp)
                # print(p_samp,'==>', sample_i)

                fill_position = self.dim_window*self.dim_in + col_i
                p_samp[0, fill_position] = sample_i
                # print('==>sample_i', sample_i)

            gen_buff.append(p_samp[0, self.dim_window*self.dim_in:])
            p_samp = torch.cat((p_samp[:, self.dim_in:], torch.zeros(1, self.dim_in)), 1)
            # print(p_samp)
        gen_buff = torch.cat(gen_buff, 0).view(-1, self.dim_in)
        # print(gen_buff)
        # print(gen_buff.shape)
            
        return gen_buff