import dgl
import torch
import scipy.sparse as sp
import numpy as np
import pandas as pd
import itertools
import networkx as nx
from utils import eids_split, remove_all_edges_of_etype, get_num_nodes_dict

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


def edge_label_creation(ecoli1_gold,edge_list):

   edge_df = pd.DataFrame(edge_list, columns =['source', 'target'])
   ecoli1_gold[0] = ecoli1_gold[0].str.replace('G', '')
   ecoli1_gold[1] = ecoli1_gold[1].str.replace('G', '')
   ecoli1_gold= ecoli1_gold.astype(int)
   ecoli1_gold[0] = ecoli1_gold[0] - 1
   ecoli1_gold[1] = ecoli1_gold[1] - 1

   edge_df['edge'] = 0
   for i in range(ecoli1_gold.shape[0]):
         r = ecoli1_gold.iat[i,0]
         c = ecoli1_gold.iat[i,1]
         idx= edge_df.loc[(edge_df['source'] == r) & (edge_df['target'] == c)].index
         edge_df.loc[idx,'edge']=ecoli1_gold.iat[i,2]
   return edge_df

def convert_grn_to_dgl_graph(file_hetero,file_null,file_traject,file_gold):
    """
    GRN 데이터를 DGL 그래프 형태로 변환하는 함수

    Parameters:
    ----------
    node_file : str
        노드 특징이 포함된 파일 경로
    edge_file : str
        엣지 데이터가 포함된 파일 경로
    gold_file : str
        라벨링된 표준(Gold Standard) 파일 경로

    Returns:
    ----------
    dgl_graph : dgl.DGLGraph
        DGL 그래프 객체
    """
    default_path="./data/DREAM4/DREAM4_InSilico_Size100/"#+folder_name+"/"+folder_name+"/"
    default_goldpath="./data/DREAM4/gold_std/"

    # Load data
    hetero = pd.read_csv(default_path + file_hetero, sep='\t')
    null = pd.read_csv(default_path + file_null, sep='\t')
    traject = pd.read_csv(default_path + file_traject, sep='\t')
    gold = pd.read_csv(default_goldpath + file_gold, sep='\t', header=None)

    # Extract wildtype values
    wildtype_vals = hetero.loc[1, :].values.tolist()
    hetero['id'] = hetero.index
    
    # Create node features
    node_features = hetero[['id']]
    node_features['wildtype'] = wildtype_vals

    traj = traject.T.iloc[1:, 1:]
    traj = traj.reset_index()
    node_features = pd.concat([node_features, traj], axis=1)
    node_features = node_features.drop(['index'], axis=1)

    # Extract edge features and edge labels
    edge_list = list(itertools.product(node_features["id"], repeat=2))
    edge_lab = edge_label_creation(gold, edge_list)

    null = null.iloc[1:, :].reset_index(drop=True)
    null_list = null.values.flatten()

    hetero = hetero.iloc[1:, :].reset_index(drop=True).drop(['id'], axis=1)
    hetero_list = hetero.values.flatten()

    edge_lab.columns = ['s', 'd', 'edge']
    edge_lab = edge_lab.iloc[100:].reset_index(drop=True)
    edge_lab['KO'] = null_list
    edge_lab['KD'] = hetero_list
    edge_lab = edge_lab[edge_lab['edge'] == 1]

    # Extract source, destination, and edge attributes
    src = edge_lab["s"].tolist()
    dst = edge_lab['d'].tolist()
    KO = edge_lab["KO"].tolist()
    KD = edge_lab["KD"].tolist()

    # Create DGL graph for basic_data
    g_basic = dgl.graph((src, dst))
    id_tensor = torch.tensor(node_features['id'].tolist(), dtype=torch.float32).view(-1, 1)
    wildtype_tensor = torch.tensor(node_features['wildtype'].tolist(), dtype=torch.float32).view(-1, 1)
    g_basic.ndata['id'] = id_tensor
    g_basic.ndata['wildtype'] = wildtype_tensor
    g_basic.edata['KO'] = torch.tensor(KO, dtype=torch.float32).view(-1, 1)
    g_basic.edata['KD'] = torch.tensor(KD, dtype=torch.float32).view(-1, 1)

    # Create DGL graph for basic_TS_data
    g_basic_TS = g_basic.clone()
    traj_tensor = torch.tensor(node_features.iloc[:, 2:].values, dtype=torch.float32)
    g_basic_TS.ndata['trajectory'] = traj_tensor

    # Create DGL graph for basic_aug_data
    G = g_basic.to_networkx().to_undirected()
    pagerank = nx.pagerank(G)
    clustering_coef = nx.clustering(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=50)
    degree = dict(G.degree())

    pagerank_tensor = torch.tensor([pagerank[i] for i in range(len(pagerank))], dtype=torch.float32).view(-1, 1)
    clustering_tensor = torch.tensor([clustering_coef[i] for i in range(len(clustering_coef))], dtype=torch.float32).view(-1, 1)
    betweenness_tensor = torch.tensor([betweenness_centrality[i] for i in range(len(betweenness_centrality))], dtype=torch.float32).view(-1, 1)
    degree_tensor = torch.tensor([degree[i] for i in range(len(degree))], dtype=torch.float32).view(-1, 1)

    g_basic_aug = g_basic.clone()
    g_basic_aug.ndata['pagerank'] = pagerank_tensor
    g_basic_aug.ndata['clustering_coef'] = clustering_tensor
    g_basic_aug.ndata['betweenness'] = betweenness_tensor
    g_basic_aug.ndata['degree'] = degree_tensor

    # Create DGL graph for basic_TS_aug_data
    g_basic_TS_aug = g_basic_TS.clone()
    g_basic_TS_aug.ndata['pagerank'] = pagerank_tensor
    g_basic_TS_aug.ndata['clustering_coef'] = clustering_tensor
    g_basic_TS_aug.ndata['betweenness'] = betweenness_tensor
    g_basic_TS_aug.ndata['degree'] = degree_tensor

    print("Basic Graph:", g_basic)
    print("Basic TS Graph:", g_basic_TS)
    print("Basic Augmented Graph:", g_basic_aug)
    print("Basic TS Augmented Graph:", g_basic_TS_aug)

    return g_basic, g_basic_TS, g_basic_aug, g_basic_TS_aug

# Example usage:
# g_basic, g_basic_TS, g_basic_aug, g_basic_TS_aug = data_preprocessing_dgl('folder_name', 'file_hetero.tsv', 'file_null.tsv', 'file_traject.tsv', 'file_gold.tsv')


def process_data(g, 
                 val_ratio, 
                 test_ratio,
                 src_ntype = 'author', 
                 tgt_ntype = 'paper',
                 pred_etype = 'likes',
                 neg='pred_etype_neg'):
    '''
    Parameters
    ----------
    g : dgl graph
    
    val_ratio : float
    
    test_ratio : float
    
    src_ntype: string
        source node type
    
    tgt_ntype: string
        target node type

    neg: string
        One of ['pred_etype_neg', 'src_tgt_neg'], different negative sampling modes. See below.
    
    Returns
    ----------
    mp_g: 
        graph for message passing.
    
    graphs containing positive edges and negative edges for train, valid, and test
    '''
    
    u, v = g.edges(etype=pred_etype)
    src_N = g.num_nodes(src_ntype)
    tgt_N = g.num_nodes(tgt_ntype)

    M = u.shape[0] # number of directed edges
    eids = torch.arange(M)
    train_pos_eids, val_pos_eids, test_pos_eids = eids_split(eids, val_ratio, test_ratio)

    train_pos_u, train_pos_v = u[train_pos_eids], v[train_pos_eids]
    val_pos_u, val_pos_v = u[val_pos_eids], v[val_pos_eids]
    test_pos_u, test_pos_v = u[test_pos_eids], v[test_pos_eids]

    if neg == 'pred_etype_neg':
        # Edges not in pred_etype as negative edges
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(src_N, tgt_N))
        adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)
    elif neg == 'src_tgt_neg':
        # Edges not connecting src and tgt as negative edges
        
        # Collect all edges between the src and tgt
        src_tgt_indices = []
        for etype in g.canonical_etypes:
            if etype[0] == src_ntype and etype[2] == tgt_ntype:
                adj = g.adj(etype=etype)
                src_tgt_index = adj.coalesce().indices()        
                src_tgt_indices += [src_tgt_index]
        src_tgt_u, src_tgt_v = torch.cat(src_tgt_indices, dim=1)

        # Find all negative edges that are not in src_tgt_indices
        adj = sp.coo_matrix((np.ones(len(src_tgt_u)), (src_tgt_u.numpy(), src_tgt_v.numpy())), shape=(src_N, tgt_N))
        adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)
    else:
        raise ValueError('Unknow negative argument')
        
    neg_eids = np.random.choice(neg_u.shape[0], min(neg_u.shape[0], M), replace=False)
    train_neg_eids, val_neg_eids, test_neg_eids = eids_split(torch.from_numpy(neg_eids), val_ratio, test_ratio)

    # train_neg_u, train_neg_v = neg_u[train_neg_eids], neg_v[train_neg_eids]
    # val_neg_u, val_neg_v = neg_u[val_neg_eids], neg_v[val_neg_eids]
    # test_neg_u, test_neg_v = neg_u[test_neg_eids], neg_v[test_neg_eids]

    # Avoid losing dimension in single number slicing
    train_neg_u, train_neg_v = np.take(neg_u, train_neg_eids), np.take(neg_v, train_neg_eids)
    val_neg_u, val_neg_v = np.take(neg_u, val_neg_eids),np.take(neg_v, val_neg_eids)
    test_neg_u, test_neg_v = np.take(neg_u, test_neg_eids), np.take(neg_v, test_neg_eids)
    
    # Construct graphs
    pred_can_etype = (src_ntype, pred_etype, tgt_ntype)
    num_nodes_dict = get_num_nodes_dict(g)
    
    train_pos_g = dgl.heterograph({pred_can_etype: (train_pos_u, train_pos_v)}, num_nodes_dict)
    train_neg_g = dgl.heterograph({pred_can_etype: (train_neg_u, train_neg_v)}, num_nodes_dict)
    val_pos_g = dgl.heterograph({pred_can_etype: (val_pos_u, val_pos_v)}, num_nodes_dict)
    val_neg_g = dgl.heterograph({pred_can_etype: (val_neg_u, val_neg_v)}, num_nodes_dict)
    test_pos_g = dgl.heterograph({pred_can_etype: (test_pos_u, test_pos_v)}, num_nodes_dict)

    test_neg_g = dgl.heterograph({pred_can_etype: (test_neg_u, test_neg_v)}, num_nodes_dict)
    
    mp_g = remove_all_edges_of_etype(g, pred_etype) # Remove pred_etype edges but keep nodes
    return mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g


def load_dataset(dataset_dir, dataset_name, val_ratio, test_ratio):
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
        
    pred_pair_to_edge_labels : dict
        key=((source node type, source node id), (target node type, target node id))
        value=dict, {cannonical edge type: (source node ids, target node ids)}
        
    pred_pair_to_path_labels : dict 
        key=((source node type, source node id), (target node type, target node id))
        value=list of lists, each list contains (cannonical edge type, source node ids, target node ids)
    '''
    graph_saving_path = f'{dataset_dir}/{dataset_name}'
    graph_list, _ = dgl.load_graphs(graph_saving_path)
    pred_pair_to_edge_labels = torch.load(f'{graph_saving_path}_pred_pair_to_edge_labels')
    pred_pair_to_path_labels = torch.load(f'{graph_saving_path}_pred_pair_to_path_labels')
    g = graph_list[0]
    if 'synthetic' in dataset_name:
        src_ntype, tgt_ntype = 'user', 'item'
    elif 'citation' in dataset_name:
        src_ntype, tgt_ntype = 'author', 'paper'

    pred_etype = 'likes'
    neg = 'src_tgt_neg'
    processed_g = process_data(g, val_ratio, test_ratio, src_ntype, tgt_ntype, pred_etype, neg)
    return g, processed_g, pred_pair_to_edge_labels, pred_pair_to_path_labels

