import torch
import numpy as np
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm

from data_grn_processing import load_grn_dataset_dgl
from model_grn import GRNGNN, prediction_dgl
from utils import set_config_args, get_comp_g_edge_labels, get_comp_g_path_labels
from utils import src_tgt_khop_in_subgraph, eval_edge_mask_auc, eval_edge_mask_topk_path_hit


# DGL 그래프에서 feature dimension 가져오기 (feat이 아닌 모든 ndata 속성 사용)
def get_in_dim(mp_g):
    """
    DGL 그래프에서 모든 노드 feature의 총 차원을 계산하는 함수
    """
    node_feats = []
    for key in mp_g.ndata.keys():  # 모든 노드 feature 속성 확인
        feat = mp_g.ndata[key]  # 해당 feature 텐서 가져오기
        if len(feat.shape) == 2:  # (num_nodes, feature_dim) 형태일 경우만 추가
            node_feats.append(feat.shape[1])
    
    if not node_feats:
        raise ValueError("No valid node features found in graph! Check ndata.")

    return sum(node_feats)  # 모든 feature 차원을 더해서 총 in_dim 반환



parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=0)
'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='aug_citation')
parser.add_argument('--valid_ratio', type=float, default=0.2) 
parser.add_argument('--test_ratio', type=float, default=0.3)
parser.add_argument('--max_num_samples', type=int, default=-1, 
                    help='maximum number of samples to explain, for fast testing. Use all if -1')

'''
GNN args
'''
parser.add_argument('--hidden_dim_1', type=int, default=128)
parser.add_argument('--hidden_dim_2', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=32)
parser.add_argument('--saved_model_dir', type=str, default='saved_models')
parser.add_argument('--saved_model_name', type=str, default='')

'''
Explanation args
'''
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops') 
parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
                    help='directory of saved explanations')
parser.add_argument('--eval_explainer_names', nargs='+', default=['pagelink'],
                    help='name of explainers to evaluate') 
parser.add_argument('--eval_path_hit', default=False, action='store_true', 
                    help='Whether to save the explanation') 
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')




parser.add_argument('--dec', type=str, default='dot_sum', choices=['dot', 'cos', 'ele', 'cat', 'dot_sum'],
                   help='Edge predictor에서 사용할 디코딩 연산 방식')
parser.add_argument('--af_val', type=str, default='F.silu', choices=['F.silu', 'F.sigmoid', 'F.tanh'],
                   help='Edge predictor에서 사용할 활성화 함수')


args = parser.parse_args()


if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'train_eval')
    
try:
    in_dim = get_in_dim(mp_g)
except KeyError:
    raise ValueError("Graph does not contain 'feat' in node features. Ensure features are properly assigned.")


g, processed_g = load_grn_dataset_dgl(args.dataset_dir,
                                        args.dataset_name,
                                        args.valid_ratio,
                                        args.test_ratio)
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g for g in processed_g]
model = GRNGNN(in_dim, args.hidden_dim_1, args.hidden_dim_2, args.out_dim,args.dec,args.af_val,args.num_layers,args.num_epochs,args.aggr,args.var).to(device)#Net(data.num_features, data.num_features, 128, 64).to(device) #self, in_channels, hidden1_channels, hidden2_channels,out_channels

if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_model'

state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cuda')
model.load_state_dict(state)    

test_src_nids, test_tgt_nids = test_pos_g.edges()
comp_graphs = defaultdict(list)
comp_g_labels = defaultdict(list)
test_ids = range(test_src_nids.shape[0])
if args.max_num_samples > 0:
    test_ids = test_ids[:args.max_num_samples]

for i in tqdm(test_ids):
    # Get the k-hop subgraph
    src_nid, tgt_nid = test_src_nids[i], test_tgt_nids[i]
    comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids = src_tgt_khop_in_subgraph( src_nid,
                                                                                               tgt_nid,
                                                                                               mp_g,
                                                                                               args.num_hops)


    with torch.no_grad():
        pred = prediction_dgl(model, comp_g, args.af_val, args.dec)

explanation_masks = {}
for explainer in args.eval_explainer_names:
    saved_explanation_mask = f'{explainer}_{args.saved_model_name}_pred_edge_to_comp_g_edge_mask'
    saved_file = Path.cwd().joinpath(args.saved_explanation_dir, saved_explanation_mask)
    with open(saved_file, "rb") as f:
        explanation_masks[explainer] = pickle.load(f)

print('Dataset:', args.dataset_name)
for explainer in args.eval_explainer_names:
    print(explainer)
    print('-'*30)
    pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]
    
    mask_auc_list = []
    for src_tgt in comp_graphs:
        comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, = comp_graphs[src_tgt]
        comp_g_edge_mask = pred_edge_to_comp_g_edge_mask[src_tgt]
        mask_auc = eval_edge_mask_auc(comp_g_edge_mask)
        mask_auc_list += [mask_auc]
      
    avg_auc = np.mean(mask_auc_list)
    
    # Print
    np.set_printoptions(precision=4, suppress=True)
    print(f'Average Mask-AUC: {avg_auc : .4f}')
    
    print('-'*30, '\n')

if args.eval_path_hit:
    topks = [3, 5, 10, 20, 50, 100, 200]
    for explainer in args.eval_explainer_names:
        print(explainer)
        print('-'*30)
        pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]

        explainer_to_topk_path_hit = defaultdict(list)
        for src_tgt in comp_graphs:
            comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, = comp_graphs[src_tgt]
            comp_g_path_labels = comp_g_labels[src_tgt][1]
            comp_g_edge_mask_dict = pred_edge_to_comp_g_edge_mask[src_tgt]
            topk_to_path_hit = eval_edge_mask_topk_path_hit(comp_g_edge_mask_dict, comp_g_path_labels, topks)

            for topk in topk_to_path_hit:
                explainer_to_topk_path_hit[topk] += [topk_to_path_hit[topk]]        

        # Take average
        explainer_to_topk_path_hit_rate = defaultdict(list)
        for topk in explainer_to_topk_path_hit:
            metric = np.array(explainer_to_topk_path_hit[topk])
            explainer_to_topk_path_hit_rate[topk] = metric.mean(0)

        # Print
        np.set_printoptions(precision=4, suppress=True)
        for k, hr in explainer_to_topk_path_hit_rate.items():
            print(f'k: {k :3} | Path HR: {hr.item(): .4f}')

        print('-'*30, '\n')
