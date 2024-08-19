import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import sys
import time
import argparse
import torch
from rdkit import Chem
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0]) # 这是一种随机游走的策略，使得随机游走更具有随机性和完备性。
    adj = sp.coo_matrix(adj) # 将增广后的邻接矩阵 adj 转换为 COO（坐标格式）稀疏矩阵，这种格式适合进行后续的矩阵运算。
    row_sum = np.array(adj.sum(1)) # 计算增广后的邻接矩阵中每个节点的度数（出度加入度）
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)

    return (d_mat.dot(adj)).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))     # 2896条边对应的行和列
    values = torch.from_numpy(sparse_mx.data)   # 2896条边对应的边权值
    shape = torch.Size(sparse_mx.shape) # torch.Size([973, 973])

    return torch.sparse.FloatTensor(indices, values, shape)
    

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx





def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def load_file_as_Adj_matrix(Alledge,features):
    import scipy.sparse as sp
    relation_matrix = np.zeros((len(features),len(features)))
    for i, j in np.array(Alledge):
        lnc, mi = int(i), int(j)
        relation_matrix[lnc, mi] = 1
        relation_matrix[mi, lnc] = 1
    Adj = sp.csr_matrix(relation_matrix, dtype=np.float32)
    return Adj
def load_data(edgelist,node_features,node_labels):
    features = sp.csr_matrix(node_features, dtype=np.float32) # 把是0的给干掉
    idx_train = range(300)
    idx_val = range(300, 400)
    idx_test = range(400, int(node_features.shape[0]))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.array(node_labels))
    adj = load_file_as_Adj_matrix(edgelist,node_features)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

def load_data1(path="../data", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum() / 2))

    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test

def graph_decompose(adj,graph_name,k,metis_p,strategy="edge"):
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","number" (depending on metis preprocessing) 
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed subgraphs
    '''
    print("Skeleton:",metis_p)
    print("Strategy:",strategy)
    g,g_rest,edges_rest,gs=get_graph_skeleton(adj,graph_name,k,metis_p)
    gs=allocate_edges(g_rest,edges_rest, gs, strategy)
       
    re=[]       
   
    edge_num_avg=0
    compo_num_avg=0
    print("Subgraph information:")
    for i in range(k):
        nodes_num=gs[i].number_of_nodes()
        edge_num=gs[i].number_of_edges()
        compo_num=nx.number_connected_components(gs[i])
        print("\t",nodes_num,edge_num,compo_num)
        edge_num_avg+=edge_num
        compo_num_avg+=compo_num
        re.append(nx.to_scipy_sparse_matrix(gs[i])) 
        
    edge_share=set(sort_edge(gs[0].edges()))
    for i in range(k):        
        edge_share&=set(sort_edge(gs[i].edges()))
        
    print("\tShared edge number is: %d"%len(edge_share))
    print("\tAverage edge number:",edge_num_avg/k) 
    print("\tAverage connected component number:",compo_num_avg/k)
    print("\n"+"-"*70+"\n")
    return re

def sort_edge(edges):
    edges=list(edges)
    for i in range(len(edges)):
        u=edges[i][0]
        v=edges[i][1]
        if u > v:
            edges[i]=(v,u)
    return edges

def get_graph_skeleton(adj,graph_name,k,metis_p): 
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","k" 
    Output:
        g:the original graph
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
    '''
    g=nx.from_numpy_matrix(adj.todense())
    num_nodes=g.number_of_nodes()
    print("Original nodes number:",num_nodes)
    num_edges=g.number_of_edges()
    print("Original edges number:",num_edges)  
    print("Original connected components number:",nx.number_connected_components(g),"\n")    
    
    g_dic=dict()
    
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()] 
            
    gs=[nx.Graph() for i in range(k)]
    for i in range(k):
        gs[i].add_nodes_from([i for i in range(num_nodes)])
    
    if metis_p=="no_skeleton":
        g_rest=g
        edges_rest=list(g_rest.edges())
    else:    
        if metis_p=="all_skeleton":
            graph_cut=g
        else:
            f=open("metis_file/"+graph_name+".graph.part.%s"%metis_p,'r')
            cluster=dict()  
            i=0
            for lines in f:
                cluster[i]=eval(lines.strip("\n"))
                i+=1
           
            graph_cut=nx.Graph()
            graph_cut.add_nodes_from([i for i in range(num_nodes)])  
            
            for v in range(num_nodes):
                v_class=cluster[v]
                for u in g_dic[v]:
                    if cluster[u]==v_class:
                        graph_cut.add_edge(v,u)
            
        subgs=list(nx.connected_component_subgraphs(graph_cut))
        print("After Metis,connected component number:",len(subgs))
        
                
        for i in range(k):
            for subg in subgs:
                T=get_spanning_tree(subg)
                gs[i].add_edges_from(T)
        
        edge_set_share=set(sort_edge(gs[0].edges()))
        for i in range(k):
            edge_set_share&=set(sort_edge(gs[i].edges()))
        edge_set_total=set(sort_edge(g.edges()))
        edge_set_rest=edge_set_total-edge_set_share   
        edges_rest=list(edge_set_rest)
        g_rest=nx.Graph()
        g_rest.add_nodes_from([i for i in range(num_nodes)])
        g_rest.add_edges_from(edges_rest)
       
          
    print("Skeleton information:")
    for i in range(k):
        print("\t",gs[i].number_of_nodes(),gs[i].number_of_edges(),nx.number_connected_components(gs[i])) 
        
    edge_set_share=set(sort_edge(gs[0].edges()))
    for i in range(k):
        edge_set_share&=set(sort_edge(gs[i].edges()))
    print("\tShared edge number is: %d\n"%len(edge_set_share))
    
    return g,g_rest,edges_rest,gs

def get_spanning_tree(g):
    '''
    Input:Graph
    Output:list of the edges in spanning tree
    '''
    g_dic=dict()
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()]
        np.random.shuffle(g_dic[v])
    flag_dic=dict()
    if g.number_of_nodes() ==1:
        return []
    gnodes=np.array(g.nodes)
    np.random.shuffle(gnodes)
    
    for v in gnodes:
        flag_dic[v]=0
    
    current_path=[]
    
    def dfs(u):
        stack=[u]
        current_node=u
        flag_dic[u]=1
        while len(current_path)!=(len(gnodes)-1):
            pop_flag=1
            for v in g_dic[current_node]:
                if flag_dic[v]==0:
                    flag_dic[v]=1
                    current_path.append((current_node,v))  
                    stack.append(v)
                    current_node=v
                    pop_flag=0
                    break
            if pop_flag:
                stack.pop()
                current_node=stack[-1]     
    dfs(gnodes[0])        
    return current_path

def allocate_edges(g_rest,edges_rest, gs, strategy):
    '''
    Input:
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed graphs after allocating rest edges
    '''
    k=len(gs)
    if strategy=="edge":  
        print("Allocate the rest edges randomly and averagely.")
        np.random.shuffle(edges_rest)
        t=int(len(edges_rest)/k)
        
        for i in range(k):       
            if i == k-1:
                gs[i].add_edges_from(edges_rest[t*i:])
            else:
                gs[i].add_edges_from(edges_rest[t*i:t*(i+1)])        
        return gs
    
    elif strategy=="node":
        print("Allocate the edges of each nodes randomly and averagely.")
        g_dic=dict()    
        for v,nb in g_rest.adjacency():
            g_dic[v]=[u[0] for u in nb.items()]
            np.random.shuffle(g_dic[v])
        
        def sample_neighbors(nb_ls,k):
            np.random.shuffle(nb_ls)
            ans=[]
            for i in range(k):
                ans.append([])
            if len(nb_ls) == 0:
                return ans
            if len(nb_ls) > k:
                t=int(len(nb_ls)/k)
                for i in range(k):
                    ans[i]+=nb_ls[i*t:(i+1)*t]
                nb_ls=nb_ls[k*t:]
            '''
            if len(nb_ls)>0:
                for i in range(k):
                    ans[i].append(nb_ls[i%len(nb_ls)])
            '''
            
            
            if len(nb_ls)>0:
                for i in range(len(nb_ls)):
                    ans[i].append(nb_ls[i])
            
            np.random.shuffle(ans)
            return ans
        
        for v,nb in g_dic.items():
            ls=np.array(sample_neighbors(nb,k))
            for i in range(k):
                gs[i].add_edges_from([(v,j) for j in ls[i]])
        
        return gs
    
from scipy.sparse import coo_matrix
def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data

def get_syn_sim(A, seq_sim, str_sim, mode):
    """

    :param A:
    :param seq_sim:
    :param str_sim:
    :param mode: 0 = GIP kernel sim
    :return:
    """
    GIP_c_sim = GIP_kernel(A) 
    GIP_d_sim = GIP_kernel(A.T)


    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    syn_drug = np.zeros((A.shape[0], A.shape[0]))
    syn_circ = np.zeros((A.shape[1], A.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if seq_sim[i, j] == 0:
                syn_drug[i, j] = GIP_c_sim[i, j]
            else:
                syn_drug[i, j] = (GIP_c_sim[i, j] + seq_sim[i, j]) / 2

    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            if str_sim[i, j] == 0:
                syn_circ[i, j] = GIP_d_sim[i, j]
            else:
                syn_circ[i, j] = (GIP_d_sim[i, j] + str_sim[i, j]) / 2
    return syn_drug, syn_circ

def GIP_kernel(asso_drug_cir):
    nc = asso_drug_cir.shape[0]
    matrix = np.zeros((nc, nc))
    r = getGosiR(asso_drug_cir)
    for i in range(nc):
        for j in range(nc):
            temp_up = np.square(np.linalg.norm(asso_drug_cir[i, :] - asso_drug_cir[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

def getGosiR(asso_drug_cir):
    nc = asso_drug_cir.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(asso_drug_cir[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def GIP_kernel(asso_drug_cir):
    nc = asso_drug_cir.shape[0]
    matrix = np.zeros((nc, nc))
    r = getGosiR(asso_drug_cir)
    for i in range(nc):
        for j in range(nc):
            temp_up = np.square(np.linalg.norm(asso_drug_cir[i, :] - asso_drug_cir[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = 1 # matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = 1 # matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)

def k_adges(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    res_1 = []
    res_2 = []
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = 1 # matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = 1 # matrix[idx_sort[i, :k + 1], i]
        for j in idx_sort[i, :k + 1]:
            res_1.append(i)
            res_2.append(j)
            res_1.append(j)
            res_2.append(i)
            res_2.append(i)
            res_1.append(i)
    return res_1,res_2

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

def accuracy(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    assert labels.dim() == 1 and outputs.dim() == 1
    outputs = outputs.ge(0.50).type(torch.int32)
    labels = labels.type(torch.int32)
    corrects = (1 - (outputs ^ labels)).type(torch.int32)
    if labels.size() == 0:
        return np.nan
    return corrects.sum().item() / labels.size()[0]


def precision(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.50).type(torch.int32).detach().cpu().numpy()

    return precision_score(labels, outputs)


def recall(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.50).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs)


def specificity(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.50).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs, pos_label=0)


def f1(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    return (precision(outputs, labels) + recall(outputs, labels)) / 2


def mcc(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    assert labels.dim() == 1 and outputs.dim() == 1
    outputs = outputs.ge(0.50).type(torch.int32)
    labels = labels.type(torch.int32)
    true_pos = (outputs * labels).sum()
    true_neg = ((1 - outputs) * (1 - labels)).sum()
    false_pos = (outputs * (1 - labels)).sum()
    false_neg = ((1 - outputs) * labels).sum()
    numerator = true_pos * true_neg - false_pos * false_neg
    deno_2 = outputs.sum() * (1 - outputs).sum() * labels.sum() * (1 - labels).sum()
    if deno_2 == 0:
        return np.nan
    return (numerator / (deno_2.type(torch.float32).sqrt())).item()


def auc(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return roc_auc_score(labels, outputs)

def aupr(outputs, labels):
    outputs=torch.from_numpy(np.array(outputs))
    labels=torch.from_numpy(np.array(labels))
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return average_precision_score(labels, outputs)
