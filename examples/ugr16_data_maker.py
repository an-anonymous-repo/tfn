import os
import sys
sys.path.insert(0, './../')

from models.synthesizer import TableFlowSynthesizer, TableFlowTransformer, CustomDatasetFromCSV
from models.network_traffic_transformer import NetworkTrafficTransformer
import numpy as np
import logging
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from argparse import ArgumentParser

def prepare_rawdata(args):
    count = 0
    ntt = NetworkTrafficTransformer()
    tft = TableFlowTransformer('data_ugr16/training.csv')
    for f in glob.glob('data_ugr16/day1_data/*.csv'):
        print('making train for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        tft.push_back(df, agg=args.n_agg, transformer=ntt)
        count += 1
    print(count)

    count = 0
    tft = TableFlowTransformer('data_ugr16/testing.csv')
    for f in glob.glob('data_ugr16/day2_data/*.csv'):
        print('making test for', f)
        this_ip = f.split("_")[-1][:-4]
        df = pd.read_csv(f)
        tft.push_back(df, agg=args.n_agg, transformer=ntt)
        count += 1
    print(count)

########################################################################
# training
########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def runner_train(args, train_file):
    print('='*25+'start loading train data'+'='*25)
    train_from_csv = CustomDatasetFromCSV(train_file, args.n_agg+1, args.n_col)
    train_loader = torch.utils.data.DataLoader(dataset=train_from_csv, batch_size=args.batch_size, shuffle=True, \
            num_workers=16, pin_memory=True)
    tfs = TableFlowSynthesizer(dim_in=args.n_col, dim_window=args.n_agg, 
            discrete_columns=[[11,12], [13, 14, 15]], learning_mode=args.learning_mode)
    
    tfs.batch_fit(train_loader, epochs=args.train_epochs)
    

    print('='*25+'start training'+'='*25) 
    train_loss = []

    
            

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--n-agg", type=int, default=5)
    argparser.add_argument("--n-col", type=int, default=16)
    argparser.add_argument("--batch-size", type=int, default=512)
    argparser.add_argument("--train-epochs", type=int, default=1000)
    argparser.add_argument("--loss-file", type=str, default='gen_ugr16/train_loss.csv')
    argparser.add_argument("--learning-mode", type=str, default='B')
    args = argparser.parse_args()
    
    # prepare_rawdata(args)

    runner_train(args, 'data_ugr16/tinytrain.csv')