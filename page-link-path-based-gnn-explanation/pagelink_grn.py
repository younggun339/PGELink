import os
import torch
import argparse
import pickle
from tqdm.auto import tqdm
from pathlib import Path

from utils import set_seed, print_args, set_config_args
from data_processing import load_dataset
from model_grn import GRNGNN
from data_grn_processing import load_grn_dataset
from explainer import PaGELink


parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=0)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='Ecoli1_basic_graph')
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
Link predictor args
'''


'''
Explanation args
'''
parser.add_argument('--lr', type=float, default=0.01, help='explainer learning_rate') 
parser.add_argument('--alpha', type=float, default=1.0, help='explainer on-path edge regularizer weight') 
parser.add_argument('--beta', type=float, default=1.0, help='explainer off-path edge regularizer weight') 
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops') 
parser.add_argument('--num_epochs', type=int, default=20, help='How many epochs to learn the mask')
parser.add_argument('--num_paths', type=int, default=40, help='How many paths to generate')
parser.add_argument('--max_path_length', type=int, default=5, help='max lenght of generated paths')
parser.add_argument('--k_core', type=int, default=2, help='k for the k-core graph') 
parser.add_argument('--prune_max_degree', type=int, default=200,
                    help='prune the graph such that all nodes have degree smaller than max_degree. No prune if -1') 
parser.add_argument('--save_explanation', default=False, action='store_true', 
                    help='Whether to save the explanation')
parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
                    help='directory of saved explanations')
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

'''
'''

parser.add_argument('--dec', type=str, default='dot_sum', choices=['dot', 'cos', 'ele', 'cat', 'dot_sum'],
                   help='Edge predictor에서 사용할 디코딩 연산 방식')
parser.add_argument('--af_val', type=str, default='F.silu', choices=['F.silu', 'F.sigmoid', 'F.tanh'],
                   help='Edge predictor에서 사용할 활성화 함수')
parser.add_argument('--var', type=str, default='ChebConv', choices=['ChebConv', 'SSGConv', 'ClusterGCNConv', 'HypergraphConv'],
                   help='GNN 변형 방식 선택')
parser.add_argument('--num_layers', type=int, default=3,
                   help='GNN의 레이어 개수')
parser.add_argument('--aggr', type=str, default='sum', choices=['sum', 'add'],
                   help='operation passed to dgl.EdgePredictor')


args = parser.parse_args()

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'pagelink')  

if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_model'
    
print_args(args)
set_seed(0)

data = load_grn_dataset(args.dataset_dir, args.dataset_name)
model = GRNGNN(data.num_features, args.hidden_dim_1, args.hidden_dim_2, args.out_dim,args.dec,args.af_val,args.num_layers,args.num_epochs,args.aggr,args.var).to(device)#Net(data.num_features, data.num_features, 128, 64).to(device) #self, in_channels, hidden1_channels, hidden2_channels,out_channels

state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='gpu')
model.load_state_dict(state)  

pagelink = PaGELink(model, 
                    lr=args.lr,
                    alpha=args.alpha, 
                    beta=args.beta, 
                    num_epochs=args.num_epochs,
                    log=True).to(device)


model = train_link_predictor(model.to(device), train_data, val_data, optimizer, criterion,args.num_epochs,args.af_val,args.dec).to(device)


test_auc, precision, recall,fpr, tpr, mcc, jac_score, cohkap_score, f1, top_k = eval_link_predictor(model, test_data,args.af_val,args.dec)

test_src_nids, test_tgt_nids = test_pos_g.edges()
test_ids = range(test_src_nids.shape[0])
if args.max_num_samples > 0:
    test_ids = test_ids[:args.max_num_samples]

pred_edge_to_comp_g_edge_mask = {}
pred_edge_to_paths = {}
for i in tqdm(test_ids):
    src_nid, tgt_nid = test_src_nids[i].unsqueeze(0), test_tgt_nids[i].unsqueeze(0)
    
    with torch.no_grad():
        pred = model(src_nid, tgt_nid, mp_g).sigmoid().item() > 0.5

    if pred:
        src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
        paths, comp_g_edge_mask_dict = pagelink.explain(src_nid, 
                                                        tgt_nid, 
                                                        mp_g,
                                                        args.num_hops,
                                                        args.prune_max_degree,
                                                        args.k_core, 
                                                        args.num_paths, 
                                                        args.max_path_length,
                                                        return_mask=True)
        
        pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict 
        pred_edge_to_paths[src_tgt] = paths

if args.save_explanation:
    if not os.path.exists(args.saved_explanation_dir):
        os.makedirs(args.saved_explanation_dir)
        
    saved_edge_explanation_file = f'pagelink_{args.saved_model_name}_pred_edge_to_comp_g_edge_mask'
    saved_path_explanation_file = f'pagelink_{args.saved_model_name}_pred_edge_to_paths'
    pred_edge_to_comp_g_edge_mask = {edge: {k: v.cpu() for k, v in mask.items()} for edge, mask in pred_edge_to_comp_g_edge_mask.items()}

    saved_edge_explanation_path = Path.cwd().joinpath(args.saved_explanation_dir, saved_edge_explanation_file)
    with open(saved_edge_explanation_path, "wb") as f:
        pickle.dump(pred_edge_to_comp_g_edge_mask, f)

    saved_path_explanation_path = Path.cwd().joinpath(args.saved_explanation_dir, saved_path_explanation_file)
    with open(saved_path_explanation_path, "wb") as f:
        pickle.dump(pred_edge_to_paths, f)
