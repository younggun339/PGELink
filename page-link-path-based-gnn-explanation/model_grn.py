import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import HeteroEmbedding, EdgePredictor

#################################################

from torch_geometric.data import Data
from torch_geometric.utils import convert
import pandas as pd
import torch
import torch_geometric.transforms as T
#import tensorflow as tf
import itertools
import numpy as np
from torch.nn import Linear
import seaborn as sns
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import HeteroData

from torch_geometric.utils import negative_sampling

from torch_geometric.nn import ChebConv
from torch_geometric.nn import HypergraphConv

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

############################
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows=999
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

###############################

class GRNGNN(nn.Module):

    def __init__(self, in_channels, hidden1_channels, hidden2_channels,out_channels,dec,af_val,num_layers,epoch,aggr,var):

        super().__init__()
        self.convs = nn.ModuleList()
        self.af_val = af_val
        self.edge_label_index = None  # 기본값 설정
        if(var=="ChebConv"):
          self.convs.append(eval(var)(in_channels, hidden1_channels,aggr=aggr,K=3))
          for _ in range(num_layers - 2):
              self.convs.append(eval(var)(hidden1_channels, hidden1_channels,aggr=aggr,K=3))
          self.convs.append(eval(var)(hidden1_channels, out_channels,aggr=aggr,K=3))
        else:
          self.convs.append(eval(var)(in_channels, hidden1_channels,aggr=aggr))
          for _ in range(num_layers - 2):
              self.convs.append(eval(var)(hidden1_channels, hidden1_channels,aggr=aggr))
          self.convs.append(eval(var)(hidden1_channels, out_channels,aggr=aggr))
        
    def set_edge_label_index(self, edge_label_index):
        self.edge_label_index = edge_label_index

    def encode(self, x, edge_index,af_val):
        prev_x = None
        x = x.to(device)
        edge_index = edge_index.to(device)   

        for i,conv in enumerate(self.convs[:-1]):
            prev_x = x
            x = conv(x, edge_index)
            if i > 0:
              x = x + prev_x
            x = eval(af_val)(x)
            # x = F.dropout(x, p=.2, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


    def decode(self, z, edge_label_index,dec):
        '''TYPE 1 - Multiplying and adding node embeddings'''
        '''TYPE 2 - Cosine similarity'''
        '''TYPE 2 - Neural Network'''
        if(dec=="dot_sum") :
          return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1)
        else:
          cos = torch.nn.CosineSimilarity(dim=1)
          output = cos(z[edge_label_index[0]].float(), z[edge_label_index[1]].float())
          return output
        
    def forward(self, x, edge_index, dec=None, **kwargs):
        if self.edge_label_index is None:
            raise ValueError("edge_label_index must be set before calling forward.")
        x = x.to(device)
        edge_index = edge_index.to(device)

        # encode 단계
        z = self.encode(x, edge_index, self.af_val)
        print(f"Net forward x.shape : {x.shape}")
        print(f"Net forward edge_index.shape : {edge_index.shape}")
        # decode 단계
        out = self.decode(z, self.edge_label_index, dec)
        print(f"Net forward z : {z.shape}")
        print(f"Net forward self.edge_label_index.shape : {self.edge_label_index.shape}")
        return out



def train_link_predictor(
    model, train_data, val_data, optimizer, criterion, n_epochs,af_val,dec
):

    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        #z = model.encode(train_data.x, train_data.edge_index,af_val)
        positive_num = train_data.edge_index.shape[1]
        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            method='sparse'
        ).to(device)
        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        ).to(device)
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0).to(device)

        z = model.encode(train_data.x, train_data.edge_index,af_val)
        out1 = model.decode(z, edge_label_index,dec)
        # decode 단계
        out = out1.view(-1)
        loss = criterion(out, edge_label.float())
        loss.backward()
        optimizer.step()

        val_auc,precision, recall,fpr, tpr, mcc, jac_score, cohkap_score, f1, top_k = eval_link_predictor(model, val_data,af_val,dec)
    
    print(f"train_link_predictor x.shape : {train_data.x.shape}")
    print(f"train_link_predictor edge_index.shape : {train_data.edge_index.shape}")
    print(f"train_link_predictor z : {z.shape}")
    print(f"train_link_predictor edge_label_index.shape : {edge_label_index.shape}")
    return model



@torch.no_grad()
def eval_link_predictor(model, data,af_val,dec):
    model.set_edge_label_index(data.edge_label_index)
    model.eval()
    z = model.encode(data.x.to(device), data.edge_index.to(device), af_val)
    out = model.decode(z, data.edge_label_index.to(device), dec)#.view(-1)
    actual = data.edge_label.cpu().numpy()
    auc = roc_auc_score(actual, out.cpu().numpy())
    fpr, tpr, _ = roc_curve(actual, out.cpu().numpy())
    precision, recall, thresholds = precision_recall_curve(actual, out.cpu().numpy())

    pred =out.cpu().numpy()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    mcc = matthews_corrcoef(actual, pred)
    jac_score = jaccard_score(actual, pred)
    cohkap_score = cohen_kappa_score(actual, pred)
    f1 = f1_score(actual, pred)
    top_k = top_k_accuracy_score(actual, pred, k=1)

    return auc,precision, recall,fpr, tpr, mcc, jac_score, cohkap_score, f1, top_k

@torch.no_grad()
def prediction(model, data,af_val,dec):
    model.eval()
    z = model.encode(data.x.to(device), data.edge_index.to(device), af_val)
    out = model.decode(z, data.edge_label_index.to(device), dec)#.view(-1)
    pred =out.cpu().numpy()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    print(pred)
    return pred



