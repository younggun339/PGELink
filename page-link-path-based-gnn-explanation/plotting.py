import networkx as nx
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt



def visualize_explanation_grn(
        computation_graph,
        edge_index,
        edge_mask,
        source_node,
        target_node,
        node_mask = None,
        seed = 0,
        figsize = (10, 10)
        ):
    
    fig, ax = plt.subplots(figsize=figsize)

    nodelist = list(computation_graph.nodes())
    edgelist = list(computation_graph.edges())


    edge_src, edge_dst = edge_mask.edges()  # 두 개의 Tensor 반환
    exp_edgelist = list(zip(edge_src.tolist(), edge_dst.tolist()))


    nodecolors = ['white' if n not in [source_node, target_node] else 'green' for n in nodelist]
    # 기본 엣지 색상은 검은색, exp_g에 포함된 엣지는 파란색
    edgecolors = ['blue' if (s, t) in exp_edgelist or (t, s) in exp_edgelist else 'black' 
              for s, t in edgelist]

    pos = nx.spring_layout(computation_graph, seed=seed)
    if node_mask is None:
        nx.draw_networkx_nodes(computation_graph, pos, node_color=nodecolors, edgecolors='black')
    else:
        nodealphas = [node_mask[n] for n in computation_graph.nodes()]
        nx.draw_networkx_nodes(computation_graph, pos, node_color=nodecolors, edgecolors='black', alpha=nodealphas)
    nx.draw_networkx_labels(computation_graph, pos)
    nx.draw_networkx_edges(computation_graph, 
                        pos, 
                        edge_color=edgecolors, 
                        edgelist=edgelist, 
                        width=3
                        )
    plt.title(f"{source_node}->{target_node}")
    sb.despine(left=True, bottom=True)
    plt.tight_layout()

    return plt.gcf()

def visualize_explanation(
        computation_graph,
        edge_index,
        edge_mask,
        source_node,
        target_node,
        node_mask = None,
        seed = 0,
        figsize = (10, 10)
        ):
    
    fig, ax = plt.subplots(figsize=figsize)

    nodelist = list(computation_graph.nodes())
    edgelist = list(computation_graph.edges())

    edge_mask_dict = {e: [] for e in edgelist}
    for i, e in enumerate(edge_index.T):
        s, t = e.numpy()
        if (s, t) in edgelist or (t, s) in edgelist:
            edge_mask_dict[tuple(sorted((s, t)))].append(edge_mask[i])
    edge_mask_dict = {k: v[0] if np.abs(v[0])>np.abs(v[1]) else v[1] for k, v in edge_mask_dict.items()}

    edge_width = []
    for s, t in edgelist:
        edge_width.append(np.abs(edge_mask_dict[tuple(sorted((s, t)))]))
    edge_width = np.array(edge_width)

    nodecolors = ['white' if n not in [source_node, target_node] else 'green' for n in nodelist]
    edgecolors = ['black' \
                if edge_mask_dict[tuple(sorted((s, t)))]>=0.  else 'darkorange' \
                for s, t in edgelist]

    pos = nx.spring_layout(computation_graph, seed=seed)
    if node_mask is None:
        nx.draw_networkx_nodes(computation_graph, pos, node_color=nodecolors, edgecolors='black')
    else:
        nodealphas = [node_mask[n] for n in computation_graph.nodes()]
        nx.draw_networkx_nodes(computation_graph, pos, node_color=nodecolors, edgecolors='black', alpha=nodealphas)
    nx.draw_networkx_labels(computation_graph, pos)
    nx.draw_networkx_edges(computation_graph, 
                        pos, 
                        edge_color=edgecolors, 
                        edgelist=edgelist, 
                        width=3*edge_width/np.abs(edge_width).max()
                        )
    plt.title(f"{source_node}->{target_node}")
    sb.despine(left=True, bottom=True)
    plt.tight_layout()

    return plt.gcf()