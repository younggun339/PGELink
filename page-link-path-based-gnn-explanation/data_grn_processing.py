
import dgl
import torch
import scipy.sparse as sp
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import os
from utils import eids_split, remove_all_edges_of_etype, get_num_nodes_dict
from collections import Counter


def process_grn_data(g,
                     val_ratio,
                     test_ratio,
                     neg):
    
    '''
    Parameters
    ----------
    g : dgl graph
    
    val_ratio : float
    
    test_ratio : float
    
    neg: string
        One of ['pred_etype_neg', 'src_tgt_neg'], different negative sampling modes. See below.
    
    Returns
    ----------
    mp_g: 
        graph for message passing.
    
    graphs containing positive edges and negative edges for train, valid, and test
    '''
    
    u, v = g.edges()

    M = u.shape[0] # number of edges
    eids = torch.arange(M)
    train_pos_eids, val_pos_eids, test_pos_eids = eids_split(eids, val_ratio, test_ratio)

    train_pos_u, train_pos_v = u[train_pos_eids], v[train_pos_eids]
    val_pos_u, val_pos_v = u[val_pos_eids], v[val_pos_eids]
    test_pos_u, test_pos_v = u[test_pos_eids], v[test_pos_eids]

    if neg == 'pred_etype_neg':
        # Edges not in pred_etype as negative edges
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.num_nodes(), g.num_nodes()))
        adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)
    else:
        raise ValueError('Unknow negative argument')
        
    neg_eids = np.random.choice(neg_u.shape[0], min(neg_u.shape[0], M), replace=False)
    train_neg_eids, val_neg_eids, test_neg_eids = eids_split(torch.from_numpy(neg_eids), val_ratio, test_ratio)

    # Avoid losing dimension in single number slicing
    train_neg_u, train_neg_v = np.take(neg_u, train_neg_eids), np.take(neg_v, train_neg_eids)
    val_neg_u, val_neg_v = np.take(neg_u, val_neg_eids),np.take(neg_v, val_neg_eids)
    test_neg_u, test_neg_v = np.take(neg_u, test_neg_eids), np.take(neg_v, test_neg_eids)


    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.num_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.num_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())

    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())
        # Create message passing graph by removing all edges (그러나 엣지 타입의 구분이 없기때문에 동일.)
    mp_g = g

    return mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g



def load_grn_dataset(dataset_dir, dataset_name, val_ratio, test_ratio):
    '''
    Parameters
    ----------
    dataset_dir : string
        dataset directory
    
    dataset_name : string
    
    val_ratio : float
    
    test_ratio : float

    Returns:
    ----------
    g: dgl graph
        The original graph

    processed_g: tuple of seven dgl graphs
        The outputs of the function `process_data`, 
        which includes g for message passing, train, valid, and test
  
    '''
    graph_saving_path = f'{dataset_dir}/{dataset_name}'
    graph_list, _ = dgl.load_graphs(graph_saving_path)
    g = graph_list[0] # 리스트로 반환되나 실상 단일 그래프이므로.
 
    neg = 'pred_etype_neg'
    processed_g = process_grn_data(g, val_ratio, test_ratio, neg)
    return g, processed_g



