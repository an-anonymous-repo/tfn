import torch
import torch.optim as optim
import torch.nn as nn
import logging
import numpy as np
import pandas as pd
import csv
import time
import os

from models.our_nets import SingleTaskNet
from torch.utils.data.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TableFlowSynthesizer(object):
    def __init__(self, dim_in, dim_window, discrete_columns=[], categorical_columns=[],
                 learning_mode='A', arch_mode='A'):
        assert learning_mode in ['A', 'B'], "Unknown Mask Type"
        self.dim_in = dim_in
        self.dim_window = dim_window
        self.cur_epoch = 0

        #######################################################################
        # prepare for discrete columns
        #######################################################################
        self.discrete_columns = discrete_columns
        self.categorical_columns_dim = categorical_columns
        self.discrete_belong = {}
        # self.discrete_dim = {}
        for dis_col in discrete_columns:
            for sub_dis_col in dis_col:
                self.discrete_belong[sub_dis_col] = dis_col[0]
                if dis_col[0] in self.categorical_columns_dim.keys():
                    self.categorical_columns_dim[dis_col[0]] += 1
                else:
                    self.categorical_columns_dim[dis_col[0]] = 1

        self.cont_agents = []
        self.disc_agents = []
        for col_i in range(self.dim_in):
            if col_i in self.discrete_belong.keys():
                if self.discrete_belong[col_i] not in self.disc_agents:
                    self.disc_agents.append(self.discrete_belong[col_i])
            elif col_i in self.categorical_columns_dim.keys():
                self.disc_agents.append(col_i)
            else:
                self.cont_agents.append(col_i)

        #######################################################################
        # initialize models
        ####################################################################### 
        print('initing', self.dim_window, dim_in)
        if arch_mode == 'A':
            encoder_arch = None
            gmm_arch = ['gmm', 2]
            gmm_lr = 0.02
            dec_arch = ['softmax', 100]
            dec_lr = 0.02
        else:
            encoder_arch = [64, 64, 'M', 128, 128, 'M']
            gmm_arch = ['gmm', 10]
            gmm_lr = 0.001
            dec_arch = ['softmax', 100] 
            dec_lr = 0.01

        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        for col_i in self.cont_agents:
            mask_mode = None if learning_mode == 'A' else col_i 
            model_i = SingleTaskNet(dim_in=dim_in, dim_out=1,
                            dim_window=dim_window, mask_mode=mask_mode,
                            encoder_arch=encoder_arch,
                            decoder_arch=gmm_arch, model_tag=col_i)
            optim_i = optim.Adam(model_i.parameters(), lr=gmm_lr) # 0.005
            sched_i = optim.lr_scheduler.StepLR(optim_i, step_size=10, gamma=0.9)

            self.models[col_i] = model_i
            self.optimizers[col_i] = optim_i
            self.schedulers[col_i] = sched_i
        
        for col_i in self.disc_agents:
            mask_mode = None if learning_mode == 'A' else col_i 
            model_i = SingleTaskNet(dim_in=dim_in, dim_out=self.categorical_columns_dim[col_i],
                        dim_window=dim_window, mask_mode=mask_mode,
                        encoder_arch=encoder_arch, 
                        decoder_arch=dec_arch, model_tag=col_i)
            optim_i = optim.Adam(model_i.parameters(), lr=dec_lr)
            sched_i = optim.lr_scheduler.StepLR(optim_i, step_size=10, gamma=0.9)

            self.models[col_i] = model_i
            self.optimizers[col_i] = optim_i
            self.schedulers[col_i] = sched_i
        
        if device.type == 'cuda':
            for col_i in sorted(self.models.keys()):
                self.models[col_i].to(device)
        # and torch.cuda.device_count() > 1:
        #         self.models[col_i] = nn.DataParallel(self.models[col_i])
        

        self.loss_file = 'train_loss.csv' 
        with open(self.loss_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([['epoch','time']+sorted(self.models.keys())])

    def _get_category(self, x):
        return (x-1024)//100+1024 if x >= 1024 else x

    def _get_variable_i(self, y, col_i):
        if col_i in self.discrete_belong.keys():
            l = self.categorical_columns_dim[col_i]
            # print('l_len', l)
            return y[:, col_i:col_i+l]
        else:
            return y[:, col_i].view(-1, 1)
            
            
   
    def batch_fit(self, train_loader, epochs=100,):
        for epoch in range(epochs):
            start_time = time.time()
            for step, (batch_X, batch_y) in enumerate(train_loader):
                minibatch = batch_X.view(-1, self.dim_window+1, self.dim_in).to(device)
                for col_i in self.cont_agents + self.disc_agents:
                    # print('training', col_i)
                    model = self.models[col_i]
                    optimizer = self.optimizers[col_i]
                    scheduler = self.schedulers[col_i]
                    y_ = self._get_variable_i(batch_y, col_i).to(device)

                    optimizer.zero_grad()
                    loss = model.loss(minibatch, y_).mean()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            end_time = time.time()
            temp = [epoch, end_time-start_time]
            for col_i in self.cont_agents + self.disc_agents:
                temp.append(self.models[col_i].get_batch_loss())
                self._save_model(col_i, epoch, self.models[col_i])
                self.models[col_i].batch_reset()
            print(temp)
            with open(self.loss_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows([temp])

    def fit(self, X, y, epochs=100):
        # X = torch.unsqueeze(X, 0)
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
                    logger.info("Model: %d\t"%col_i + "Iter: %d\t"%i + "Loss: %.2f"%loss.data)

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
    
    def _save_model(self, name, epoch, model):
        if not os.path.exists('./saved_model'):
            os.makedirs('./saved_model')
        if not os.path.exists('./saved_model/model_%d'%name):
            os.makedirs('./saved_model/model_%d'%name)
        checkpoint = './saved_model/model_%d'%name + '/ep%d.pkl'%epoch
        torch.save(model.state_dict(), checkpoint)

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.height = height
        self.width = width

    def __getitem__(self, index):
        single_image_label = np.asarray(self.data.iloc[index]).reshape(self.height+1,self.width).astype(np.float32)[-1]
        img_as_np = np.asarray(self.data.iloc[index]).reshape(self.height+1,self.width).astype(np.float32)[:-1]
        img_as_tensor = torch.from_numpy(img_as_np)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

class TableFlowTransformer(object):
    def __init__(self, output_file, special_data=None):
        self.output_file = output_file
        self.special_data = special_data
        # self.df_naive = df
        self.X ,self.y = None, None
        f = open(output_file, "w+")
        f.close()
    
    def push_back(self, df, agg=1, transformer=None):
        df = df.dropna()
        if transformer:
            df = transformer.transfer(df)
        # print(df)
        X, y = self.agg(df, agg=agg)
        X.to_csv(self.output_file, mode='a', header=False, index=False)


    def agg(self, df, agg=None):
        if agg:
            self.X, self.y = self._agg_window(df, agg)
        return self.X, self.y

    def _get_category(self, x):
        return (x-1024)//100+1024 if x >= 1024 else x

    def _ugr16_label(self, row_list):
        new_list = row_list.copy()
        new_list[5] = self._get_category(np.rint(new_list[5] * 65536))
        new_list[6] = self._get_category(np.rint(new_list[6] * 65536))
        new_list[7] = min(np.rint(new_list[7]*255), 255)
        new_list[8] = min(np.rint(new_list[8]*255), 255)
        new_list[9] = min(np.rint(new_list[9]*255), 255)
        new_list[10] = min(np.rint(new_list[10]*255), 255)
        return new_list

    def _agg_window(self, df_naive, agg_size):
        col_num = len(df_naive.columns)
        buffer = [[0]*col_num] * agg_size
        X, y = [], []

        list_naive = df_naive.values.tolist()
        for row in list_naive:
            buffer.append(row)
            row_with_window = []
            for r in buffer[-agg_size-1:]:
                row_with_window += r
            ugr16_label_row = self._ugr16_label(row)
            #row_label = 
            X.append(row_with_window+ugr16_label_row)
            y.append(row)

        # X = torch.Tensor(X).view(-1, col_num*(agg_size+1))
        # y = torch.Tensor(y).view(-1, col_num)
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        return X, y
