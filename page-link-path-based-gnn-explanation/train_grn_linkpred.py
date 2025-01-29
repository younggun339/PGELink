
import os
import torch
import torch.nn.functional as F
import copy
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
from utils import set_seed, negative_sampling, print_args, set_config_args
from data_grn_processing import load_grn_dataset


##

from torch_geometric.data import Data
from torch_geometric.utils import convert
import torch
import pandas as pd
import torch_geometric.transforms as T
#import tensorflow as tf
import itertools
import numpy as np
import torch
from torch.nn import Linear
import seaborn as sns
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import HeteroData

from torch_geometric.utils import negative_sampling

from torch_geometric.nn import ChebConv
from torch_geometric.nn import HypergraphConv
import torch.nn.functional as F

#FOR VISUALIZING GRAPH
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import Counter
from torch_geometric.utils import to_networkx


from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import DetCurveDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


from model_grn import GRNGNN, train_link_predictor, eval_link_predictor

##

parser = argparse.ArgumentParser(description='Train a GNN-based link prediction model')
parser.add_argument('--device_id', type=int, default=0)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='Ecoli1_basic_graph')
parser.add_argument('--valid_ratio', type=float, default=0.2) 
parser.add_argument('--test_ratio', type=float, default=0.3)

'''
GNN args
'''
parser.add_argument('--hidden_dim_1', type=int, default=128)
parser.add_argument('--hidden_dim_2', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=32)

'''
Link predictor args
'''
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')
parser.add_argument('--lr', type=float, default=0.01, help='link predictor learning_rate') 
parser.add_argument('--num_epochs', type=int, default=200, help='How many epochs to train')
parser.add_argument('--eval_interval', type=int, default=1, help="Evaluate once per how many epochs")
parser.add_argument('--save_model', default=False, action='store_true', help='Whether to save the model')
parser.add_argument('--saved_model_dir', type=str, default='saved_models', help='Where to save the model')
parser.add_argument('--sample_neg_edges', default=False, action='store_true', 
                    help='If False, use fixed negative edges. If True, sample negative edges in each epoch')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

args = parser.parse_args()

    
if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'train_eval')
    
print_args(args)



''' CODE TO RUN AFTER PREPROCESSING '''
def main_run():
  auprs =0
  aucs =0

  for i in range(1) : ## 본래10, 임시로 3
    print("i is : ", i)
    split = T.RandomLinkSplit(
        num_val=args.valid_ratio,
        num_test=args.test_ratio,
        is_undirected=False,
        add_negative_train_samples=True,
        key="edge_label"
    )

    train_data, val_data, test_data = split(args.dataset)

# 데이터 디바이스 이동
    train_data.x = train_data.x.to(device)
    train_data.edge_index = train_data.edge_index.to(device)
    train_data.edge_label = train_data.edge_label.to(device)
    train_data.edge_label_index = train_data.edge_label_index.to(device)

    val_data.x = val_data.x.to(device)
    val_data.edge_index = val_data.edge_index.to(device)
    val_data.edge_label = val_data.edge_label.to(device)
    val_data.edge_label_index = val_data.edge_label_index.to(device)

    test_data.x = test_data.x.to(device)
    test_data.edge_index = test_data.edge_index.to(device)
    test_data.edge_label = test_data.edge_label.to(device)
    test_data.edge_label_index = test_data.edge_label_index.to(device)

    index=(train_data.edge_label == 1).nonzero(as_tuple=True)[0]

    ''' OLD CODE'''
    in_channels=data.num_features
    print(f"in_channels : {in_channels}")

    model = train_link_predictor(model, train_data, val_data, optimizer, criterion,args.epoch,args.af_val,args.dec).to(device)


    test_auc, precision, recall,fpr, tpr, mcc, jac_score, cohkap_score, f1, top_k = eval_link_predictor(model, test_data,args.af_val,args.dec)
    aucs = aucs+test_auc
    aupr = auc(recall, precision)
    auprs=auprs+aupr

  mean_auc = float(aucs/10)
  mean_aupr = float(auprs/10)


  list1 = [args.dec, args.af_val,args.num_layers,args.epoch, args.aggr, args.var,mean_auc, mean_aupr, mcc, jac_score,cohkap_score, f1, top_k]
  df = pd.DataFrame(list1).T
  df.columns = ["dec", "af_val","num_layers", "epoch", "aggr", "var","auc", "aupr", "mcc", "jac_score","cohkap_score", "f1", "top_k"]


  return df



data = load_grn_dataset(args.dataset_dir, args.dataset_name)

model =  GRNGNN(data.num_features, args.hidden_dim_1, args.hidden_dim_2, args.out_dim,args.dec,args.af_val,args.num_layers,args.epoch,args.aggr,args.var).to(device)#Net(data.num_features, data.num_features, 128, 64).to(device) #self, in_channels, hidden1_channels, hidden2_channels,out_channels

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)#RMSprop(params=model.parameters())#
criterion = torch.nn.BCEWithLogitsLoss()

main_run()

if args.save_model:
    output_dir = Path.cwd().joinpath(args.saved_model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), output_dir.joinpath(f"{args.dataset_name}_model.pth"))

