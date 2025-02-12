import dgl
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from utils import get_ntype_hetero_nids_to_homo_nids, get_homo_nids_to_ntype_hetero_nids, get_ntype_pairs_to_cannonical_etypes
from utils import src_tgt_khop_in_subgraph, get_neg_path_score_func, k_shortest_paths_with_max_length
from model_grn import prediction_dgl
def get_edge_mask_dict(ghomo):
    """
    동질 그래프에서 엣지 마스크를 생성하는 함수.

    Parameters
    ----------
    ghomo : dgl.DGLGraph
        동질 그래프 (homogeneous graph).

    Returns
    -------
    edge_mask : torch.nn.Parameter
        그래프의 모든 엣지에 대한 학습 가능한 마스크 (1D 텐서).
    """
    device = ghomo.device
    num_edges = ghomo.num_edges()
    num_nodes = ghomo.num_nodes()

    std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
    edge_mask = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)

    return edge_mask

def remove_edges_of_high_degree_nodes(ghomo, max_degree=10, always_preserve=[]):
    '''
    For all the nodes with degree higher than `max_degree`, 
    except nodes in `always_preserve`, remove their edges. 
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    max_degree : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    low_degree_ghomo : dgl homogeneous graph
        Pruned graph with edges of high degree nodes removed

    '''
    d = ghomo.in_degrees()
    high_degree_mask = d > max_degree
    
    # preserve nodes
    high_degree_mask[always_preserve] = False    

    high_degree_nids = ghomo.nodes()[high_degree_mask]
    u, v = ghomo.edges()
    high_degree_edge_mask = torch.isin(u, high_degree_nids) | torch.isin(v, high_degree_nids)
    high_degree_u, high_degree_v = u[high_degree_edge_mask], v[high_degree_edge_mask]
    high_degree_eids = ghomo.edge_ids(high_degree_u, high_degree_v)
    low_degree_ghomo = dgl.remove_edges(ghomo, high_degree_eids)
    
    return low_degree_ghomo


def remove_edges_except_k_core_graph(ghomo, k, always_preserve=[]):
    '''
    Find the `k`-core of `ghomo`.
    Only isolate the low degree nodes by removing theirs edges
    instead of removing the nodes, so node ids can be kept.
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    k : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    k_core_ghomo : dgl homogeneous graph
        The k-core graph
    '''
    k_core_ghomo = ghomo
    degrees = k_core_ghomo.in_degrees()
    k_core_mask = (degrees > 0) & (degrees < k)
    k_core_mask[always_preserve] = False
    
    while k_core_mask.any():
        k_core_nids = k_core_ghomo.nodes()[k_core_mask]
        
        u, v = k_core_ghomo.edges()
        k_core_edge_mask = torch.isin(u, k_core_nids) | torch.isin(v, k_core_nids)
        k_core_u, k_core_v = u[k_core_edge_mask], v[k_core_edge_mask]
        k_core_eids = k_core_ghomo.edge_ids(k_core_u, k_core_v)

        k_core_ghomo = dgl.remove_edges(k_core_ghomo, k_core_eids)
        
        degrees = k_core_ghomo.in_degrees()
        k_core_mask = (degrees > 0) & (degrees < k)
        k_core_mask[always_preserve] = False

    return k_core_ghomo

def get_eids_on_paths(paths, ghomo):
    '''
    Collect all edge ids on the paths
    
    Note: The current version is a list version. An edge may be collected multiple times
    A different version is a set version where an edge can only contribute one time 
    even it appears in multiple paths
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    Returns
    -------
    paths: list of lists
        Each list contains (source node ids, target node ids)
        
    '''
    row, col = ghomo.edges()
    eids = []
    for path in paths:
        for i in range(len(path)-1):
            eids += [((row == path[i]) & (col == path[i+1])).nonzero().item()]
            
    return torch.LongTensor(eids)

def comp_g_paths_to_paths(comp_g, comp_g_paths):
    return comp_g_paths


class PaGELink(nn.Module):
    """Path-based GNN Explanation for Heterogeneous Link Prediction (PaGELink)
    
    Some methods are adapted from the DGL GNNExplainer implementation
    https://docs.dgl.ai/en/0.8.x/_modules/dgl/nn/pytorch/explain/gnnexplainer.html#GNNExplainer
    
    Parameters
    ----------
    model : nn.Module
        The GNN-based link prediction model to explain.

        * The required arguments of its forward function are source node id, target node id,
          graph, and feature ids. The feature ids are for selecting input node features.
        * It should also optionally take an eweight argument for edge weights
          and multiply the messages by the weights during message passing.
        * The output of its forward function is the logits in (-inf, inf) for the 
          predicted link.
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    alpha1 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the sum of the edge mask.
    alpha2 : float, optional
        A higher value will make the explanation edge masks more discrete by decreasing
        the entropy of the edge mask.
    alpha : float, optional
        A higher value will make edges on high-quality paths to have higher weights
    beta : float, optional
        A higher value will make edges off high-quality paths to have lower weights
    log : bool, optional
        If True, it will log the computation process, default to True.
    """
    def __init__(self,
                 model,
                 lr=0.001,
                 num_epochs=100,
                 alpha=1.0,
                 beta=1.0,
                 log=False,
                 af_val="F.silu"):
        super(PaGELink, self).__init__()
        self.model = model
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.beta = beta
        self.log = log
        self.af_val = af_val
        
        self.all_loss = defaultdict(list)

    def _init_masks(self, ghomo):
        """Initialize the learnable edge mask.

        Parameters
        ----------
        graph : DGLGraph
            Input graph.

        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        return get_edge_mask_dict(ghomo)
    

    def _prune_graph(self, ghomo, prune_max_degree=-1, k_core=2, always_preserve=[]):
        # Prune edges by (optionally) removing edges of high degree nodes and extracting k-core
        # The pruning is computed on the homogeneous graph, i.e., ignoring node/edge types
        device = ghomo.device
        ghomo.edata['eid_before_prune'] = torch.arange(ghomo.num_edges()).to(device)
        
        if prune_max_degree > 0:
            max_degree_pruned_ghomo = remove_edges_of_high_degree_nodes(ghomo, prune_max_degree, always_preserve)
            k_core_ghomo = remove_edges_except_k_core_graph(max_degree_pruned_ghomo, k_core, always_preserve)
            
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = max_degree_pruned_ghomo
            else:
                pruned_ghomo = k_core_ghomo
        else:
            k_core_ghomo = remove_edges_except_k_core_graph(ghomo, k_core, always_preserve)
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = ghomo
            else:
                pruned_ghomo = k_core_ghomo

         # Pruning된 엣지 마스크 생성
        pruned_ghomo_eids = pruned_ghomo.edata['eid_before_prune']
        pruned_ghomo_eid_mask = torch.zeros(ghomo.num_edges(), dtype=torch.bool, device=device)
        pruned_ghomo_eid_mask[pruned_ghomo_eids] = True  # pruning되지 않은 엣지만 True
        return pruned_ghomo, pruned_ghomo_eid_mask
        
        
    def path_loss(self, src_nid, tgt_nid, g, eweights, num_paths=5):
        """Compute the path loss.

        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        g : dgl graph

        eweights : Tensor
            Edge weights with shape equals the number of edges.
            
        num_paths : int
            Number of paths to compute path loss on

        Returns
        -------
        loss : Tensor
            The path loss
        """
        neg_path_score_func = get_neg_path_score_func(g, 'eweight', [src_nid, tgt_nid])
        paths = k_shortest_paths_with_max_length(g, 
                                                 src_nid, 
                                                 tgt_nid, 
                                                 weight=neg_path_score_func, 
                                                 k=num_paths)

        eids_on_path = get_eids_on_paths(paths, g)

        if eids_on_path.nelement() > 0:
            loss_on_path = - eweights[eids_on_path].mean()
        else:
            loss_on_path = 0

        eids_off_path_mask = ~torch.isin(torch.arange(eweights.shape[0]), eids_on_path)
        if eids_off_path_mask.any():
            loss_off_path = eweights[eids_off_path_mask].mean()
        else:
            loss_off_path = 0

        loss = self.alpha * loss_on_path + self.beta * loss_off_path 

        self.all_loss['loss_on_path'] += [float(loss_on_path)]
        self.all_loss['loss_off_path'] += [float(loss_off_path)]

        return loss   

    
    def get_edge_mask(self, 
                      src_nid, 
                      tgt_nid, 
                      ghomo, 
                      feat_nids, 
                      prune_max_degree=-1,
                      k_core=2, 
                      prune_graph=True,
                      with_path_loss=True):

        """Learning the edge mask dict.   
        
        Parameters
        ----------
        see the `explain` method.
        
        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """

        self.model.eval()
        device = ghomo.device
        
        homo_src_nid = int(src_nid)
        homo_tgt_nid = int(tgt_nid)

        # Get the initial prediction.
        with torch.no_grad():
            print()
            # 모델을 사용하여 전체 그래프의 예측 수행
            pred_all = prediction_dgl(self.model, ghomo, self.af_val, "dot_sum")  

            # 특정 src_nid와 tgt_nid에 해당하는 예측 값만 선택
            edge_index = torch.stack(ghomo.edges(), dim=0).cpu().numpy()
            src_tgt_pair = np.array([src_nid.cpu().numpy(), tgt_nid.cpu().numpy()]).reshape(2, 1)

            # edge_index에서 해당하는 예측값 찾기
            mask = np.all(edge_index == src_tgt_pair, axis=0)

            score = pred_all[mask][0]   # 해당 링크의 예측 점수

            # 최종 예측값 변환
            pred = (score > 0).astype(int)



        if prune_graph:
            # Prune the graph and return a homogeneous subgraph
            ml_ghomo, pruned_ghomo_eid_mask = self._prune_graph(ghomo, prune_max_degree, k_core, [homo_src_nid, homo_tgt_nid])
        else:
            # Use the original homogeneous graph
            ml_ghomo = ghomo
            
        ml_edge_mask = self._init_masks(ml_ghomo) 
        optimizer = torch.optim.Adam([ml_edge_mask], lr=self.lr)  

        if self.log:
            pbar = tqdm(total=self.num_epochs)

        eweight_norm = 0
        EPS = 1e-3
        for e in range(self.num_epochs):    
            # 모델을 사용하여 전체 그래프 예측 수행 (그라디언트 추적 유지)
            pred_all = prediction_dgl(self.model, ml_ghomo, self.af_val, "dot_sum")  


            # 특정 src_nid와 tgt_nid에 해당하는 예측 값 선택
            edge_index = torch.stack(ml_ghomo.edges(), dim=0).cpu().numpy()
            src_tgt_pair = np.array([src_nid.cpu().numpy(), tgt_nid.cpu().numpy()]).reshape(2, 1)

            # edge_index에서 해당하는 예측값 찾기
            mask = np.all(edge_index == src_tgt_pair, axis=0)

                        # pos_pred_all이 numpy이면 텐서로 변환
            if isinstance(pred_all, np.ndarray):
                pred_all = torch.tensor(pred_all, dtype=torch.float32, device=device, requires_grad=True)


            score = pred_all[mask].to(dtype=torch.float32, device=device)[0]

            # 예측 손실 계산
            pred_loss = (-1) ** pred * torch.sigmoid(score).log()
            self.all_loss['pred_loss'] += [pred_loss.item()]

            ml_ghomo.edata['eweight'] = ml_ghomo.edata['KD'].float() + ml_ghomo.edata['KO'].float()
            # 엣지 가중치 가져오기
            ml_ghomo_eweights = ml_ghomo.edata['eweight']

            # Path loss 추가
            if with_path_loss:
                path_loss = self.path_loss(homo_src_nid, homo_tgt_nid, ml_ghomo, ml_ghomo_eweights)
            else:
                path_loss = 0

            # 최종 손실 계산
            loss = pred_loss + path_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.all_loss['total_loss'] += [loss.item()]

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

            # 동질 그래프 기반으로 엣지 마스크 초기화
        edge_mask = self._init_masks(ml_ghomo)

        if prune_graph:
            # Pruned된 엣지 제외 (동질 그래프 기준으로 필터링)
            pruned_eid_mask = pruned_ghomo_eid_mask  # 기존 etypes별 마스크가 아닌, 단일 마스크
            edge_mask = torch.full_like(pruned_eid_mask, float('-inf'), dtype=torch.float) 

            edge_mask[pruned_eid_mask] = ml_edge_mask
        else:
            edge_mask = ml_edge_mask  # 동질 그래프에서는 바로 사용 가능

        # detach() 후 반환
        return edge_mask.detach()

    def get_paths(self,
                  src_nid, 
                  tgt_nid, 
                  ghomo,
                  edge_mask,
                  num_paths=1, 
                  max_path_length=3):

        """A postprocessing step that turns the `edge_mask_dict` into actual paths.
        
        Parameters
        ----------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges

        Others: see the `explain` method.
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)
        """
        print("edge_mask shape:", edge_mask.shape)
        print("ghomo num_edges:", ghomo.num_edges())
        eweight = edge_mask.sigmoid()
        ghomo.edata['eweight'] = eweight
        print(f"eweigth : {eweight}")
        # convert ghetero to ghomo and find paths
       

        homo_src_nid = int(src_nid)
        homo_tgt_nid = int(tgt_nid)
        print(f"homo_src_nid : {homo_src_nid}")
        neg_path_score_func = get_neg_path_score_func(ghomo, 'eweight', [src_nid.item(), tgt_nid.item()])
        homo_paths = k_shortest_paths_with_max_length(ghomo, 
                                                       homo_src_nid, 
                                                       homo_tgt_nid,
                                                       weight=neg_path_score_func,
                                                       k=num_paths,
                                                       max_length=max_path_length)

        paths = [homo_paths]

        return paths
    
    def explain(self,  
                src_nid, 
                tgt_nid, 
                ghomo,
                num_hops=2,
                prune_max_degree=-1,
                k_core=2, 
                num_paths=1, 
                max_path_length=3,
                prune_graph=True,
                with_path_loss=True,
                return_mask=False):
        
        """Return a path explanation of a predicted link
        
        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        ghetero : dgl graph

        num_hops : int
            Number of hops to extract the computation graph, i.e. GNN # layers
            
        prune_max_degree : int
            If positive, prune the edges of graph nodes with degree larger than `prune_max_degree`
            If  -1, do nothing
            
        k_core : int 
            k for the the k-core graph extraction
            
        num_paths : int
            Number of paths for the postprocessing path extraction
            
        max_path_length : int
            Maximum length of paths for the postprocessing path extraction
        
        prune_graph : bool
            If true apply the max_degree and/or k-core pruning. For ablation. Default True.
            
        with_path_loss : bool
            If true include the path loss. For ablation. Default True.
            
        return_mask : bool
            If true return the edge mask in addition to the path. For AUC evaluation. Default False
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)

        (optional) edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        # Extract the computation graph (k-hop subgraph)
        (comp_g_src_nid, 
         comp_g_tgt_nid, 
         comp_g, 
         comp_g_feat_nids) = src_tgt_khop_in_subgraph(       src_nid, 
                                                             tgt_nid, 
                                                             ghomo, 
                                                             num_hops)
        # Learn the edge mask on the computation graph
        comp_g_edge_mask = self.get_edge_mask(comp_g_src_nid, 
                                                   comp_g_tgt_nid, 
                                                   comp_g, 
                                                   comp_g_feat_nids,
                                                   prune_max_degree,
                                                   k_core,
                                                   prune_graph,
                                                   with_path_loss)

        # Extract paths 
        comp_g_paths = self.get_paths(comp_g_src_nid,
                                      comp_g_tgt_nid, 
                                      comp_g, 
                                      comp_g_edge_mask, 
                                      num_paths, 
                                      max_path_length)    
        
        
        # Covert the node id in computation graph to original graph
        paths = comp_g_paths_to_paths(comp_g, comp_g_paths)
        
        if return_mask:
            # return masks for easier evaluation
            return paths, comp_g_edge_mask
        else:
            return paths 



