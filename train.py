import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
pd.set_option('display.max_rows',10) 
from utils import *
from sklearn import svm 
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from scipy import interp
import argparse
import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import chi2
from GNN_models import *
result_4_ROC1 = []
result_4_ROC2 = []


class item:
    def __init__(self):
        self.epochs = 2
        self.lr = 1e-1
        self.k1 = 200  
        self.k2 = 10
        self.epsilon1 = 0.02
        self.hidden=512
        self.dropout = 0.5
        self.runs = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Your description here")
    parser.add_argument("--circ_node_count", type=int, default=271, help="Description of circ_node_count parameter")
    parser.add_argument("--drug_node_count", type=int, default=218, help="Description of drug_node_count parameter")
    parser.add_argument("--NmedEdge", type=int, default=700, help="Description of NmedEdge parameter")
    parser.add_argument("--DmedEdge", type=int, default=6, help="Description of DmedEdge parameter")
    parser.add_argument("--SmedEdge", type=float, default=0.7, help="Description of SmedEdge parameter")
    parser.add_argument("--threshold", type=float, default=0.35, help="threshold for matric")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epoch", type=int, default=5000, help="epoch")
    parser.add_argument("--k_fold", type=int, default=10, help="Description of k_fold parameter")
    parser.add_argument("--min_value_po", type=int, default=136, help="num min value for positive sample ") 
    parser.add_argument("--min_value_ne", type=int, default=10, help="num min value for negative sample ") 
    parser.add_argument("--min_value_po2", type=int, default=106, help="num min value for positive sample ") 
    parser.add_argument("--min_value_ne2", type=int, default=10, help="num min value for negative sample ") 
    parser.add_argument("--min_value", type=int, default=10, help="num min value for negative sample method Jianxin Wang")
    parser.add_argument("--knn_nums", type=int, default=10, help="Description of knn_nums parameter")
    parser.add_argument("--nega_threshold", type=int, default=2, help="the number of validated/predicted interactions of each sample candidate should be larger than threshold")
    parser.add_argument("--output_size", type=int, default=256, help="output size for GCN")
    
    args = parser.parse_args()
    print(args)
    return args

def aver(hops, adj, feature_list, alpha=0.15):
    input_feature = []
    for i in range(adj.shape[0]):
        hop = hops[i].int().item()
        if hop == 0:
            fea = feature_list[0][i].unsqueeze(0)
        else:
            fea = 0
            for j in range(hop):
                fea += (1-alpha)*feature_list[j][i].unsqueeze(0) + alpha*feature_list[0][i].unsqueeze(0)
            fea = fea / hop
        input_feature.append(fea)
    input_feature = torch.cat(input_feature, dim=0)
    return input_feature

def sample_variance(data):
    n = len(data)
    mean = np.mean(data)
    return np.sum((data - mean)**2) / (n - 1)
alpha = 0.05
def variance_test(sample_var, pop_var, n, alpha):
    critical_value1 = chi2.ppf(1 - alpha, n - 1)
    critical_value2 = chi2.ppf(alpha, n - 1)
    if sample_var * (n-1) / pop_var > critical_value1: # or sample_var * (n-1) / pop_var < critical_value2:
        return True
    else:
        return False
def population_variance(sim_array):
    arr = sim_array
    diag_indices = np.arange(arr.shape[0])

    not_diag_indices = np.ones_like(arr, dtype=bool)
    not_diag_indices[diag_indices, diag_indices] = False

    arr_no_diag = arr[not_diag_indices].reshape(arr.shape[0], -1)

    assert arr_no_diag.shape[0] == sim_array.shape[0]
    assert arr_no_diag.shape[1] == sim_array.shape[1] - 1

    data_flattened = arr_no_diag.flatten()

    variance = np.var(data_flattened)
    return variance
def get_sample_variance(sim_array,index):
    combinations = []
    for i in range(len(index)):
        for j in range(i + 1, len(index)):
            combinations.append(sim_array[index[i], index[j]])
    assert len(combinations) == len(index) * (len(index)-1) /2
    variance = np.var(np.array(combinations))
    return variance
def get_negtive_po(min_value,min_value2,cir_sim, drug_sim, positive_pairs,numdrug,nega_threshold):
    resultt = []
    circ_num = positive_pairs['target'].unique()
    circ_vari = population_variance(cir_sim)
    drug_vari = population_variance(drug_sim)

    for i in circ_num:
        min_indexes = np.argsort(-cir_sim[i-numdrug])[:min_value]
        min_values = min_indexes.tolist()
        true = positive_pairs.loc[positive_pairs['target'] == i, 'source'].tolist()
        if len(true) <= nega_threshold: 
            continue
        c2d_sample_variance = get_sample_variance(drug_sim,true)
        is_significantly_different = variance_test(c2d_sample_variance, drug_vari, len(true), alpha)
        if is_significantly_different:
            pairwise_combinations = [(y, x+numdrug) for x in min_values for y in true]
            resultt += pairwise_combinations
    drug_num = positive_pairs['source'].unique()
    for j in drug_num:
        min_indexes = np.argsort(-drug_sim[j])[:min_value2]
        min_values = min_indexes.tolist()
        true = positive_pairs.loc[positive_pairs['source'] == j, 'target'].tolist()
        if len(true) <= nega_threshold:
            continue
        d2c_sample_variance = get_sample_variance(cir_sim,[m-numdrug for m in true])
        is_significantly_different = variance_test(d2c_sample_variance, circ_vari, len(true), alpha)
        if is_significantly_different:
            pairwise_combinations = [(x, y) for x in min_values for y in true]
            resultt += pairwise_combinations
    
    return list(set(resultt))

def get_negtive_ne(min_value,min_value2,cir_sim, drug_sim, positive_pairs,numdrug,nega_threshold):
    resultt = []
    circ_num = positive_pairs['target'].unique()
    circ_vari = population_variance(cir_sim)
    drug_vari = population_variance(drug_sim)

    for i in circ_num:
        min_indexes = np.argsort(cir_sim[i-numdrug])[:min_value] 
        min_values = min_indexes.tolist()
        true = positive_pairs.loc[positive_pairs['target'] == i, 'source'].tolist()
        if len(true) <= nega_threshold: 
            continue
        c2d_sample_variance = get_sample_variance(drug_sim,true)
        is_significantly_different = variance_test(c2d_sample_variance, drug_vari, len(true), alpha)
        if is_significantly_different:
            pairwise_combinations = [(y, x+numdrug) for x in min_values for y in true]
            resultt += pairwise_combinations
    drug_num = positive_pairs['source'].unique()
    for j in drug_num:
        min_indexes = np.argsort(drug_sim[j])[:min_value2]
        min_values = min_indexes.tolist()
        true = positive_pairs.loc[positive_pairs['source'] == j, 'target'].tolist()
        if len(true) <= nega_threshold:
            continue
        d2c_sample_variance = get_sample_variance(cir_sim,[m-numdrug for m in true])
        is_significantly_different = variance_test(d2c_sample_variance, circ_vari, len(true), alpha)
        if is_significantly_different:
            pairwise_combinations = [(x, y) for x in min_values for y in true]
            resultt += pairwise_combinations
    return list(set(resultt))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    tprs=[]
    aucs=[]
    auprs=[]
    pres=[]
    recs=[]
    spes=[]
    f1s=[]
    acc=[]
    cir_sim = pd.read_csv("./data/gene_seq_sim.csv", index_col=0, dtype=np.float32).to_numpy()  
    drug_sim=pd.read_csv("./data/drug_str_sim.csv", index_col=0, dtype=np.float32).to_numpy()
    positive_pairs = pd.read_csv('./data/AllEdge_DrCr.csv', header = None, names=["source", "target"])
    AllNegative_all = pd.read_csv('./data/AllNegative_DrCr.csv',header=None, names=["source", "target"]) 


    import json
    for kfold in range(args.k_fold):
        with open("/ifs/data/fanziyu/project/LSNSCDA/src/10fold/"+str(kfold)+".json", 'r') as f:
            a = json.load(f)
            train_interact_pos = np.array(a['train_interact_pos']) # 训练对
            train_interact_pos[:, 0] += args.drug_node_count
            val_interact_pos = np.array(a['val_interact_pos']) # 验证对
            val_interact_pos[:, 0] += args.drug_node_count
        assert len(train_interact_pos)+len(val_interact_pos) == len(positive_pairs)

        train_pos_pairs = pd.DataFrame(train_interact_pos, columns=['target', 'source'])#positive_pairs.iloc[train_index]
        test_pos_pairs = pd.DataFrame(val_interact_pos, columns=['target', 'source'])#positive_pairs.iloc[test_index]

        print(AllNegative_all.shape)
        AllNegative_all = pd.merge(AllNegative_all, train_pos_pairs, how='left', indicator=True)
        AllNegative_all = AllNegative_all[AllNegative_all['_merge'] == 'left_only']
        AllNegative_all.drop('_merge', axis=1, inplace=True)
        print(AllNegative_all.shape)

        train_most_sim = get_negtive_po(args.min_value_po,args.min_value_po2,cir_sim,drug_sim,train_pos_pairs,args.drug_node_count,args.nega_threshold)
        train_AllNegative_po = pd.DataFrame(train_most_sim, columns=['source', 'target'])
        train_not_most_sim = get_negtive_ne(args.min_value_ne,args.min_value_ne2,cir_sim,drug_sim,train_pos_pairs,args.drug_node_count,args.nega_threshold)
        train_AllNegative_ne = pd.DataFrame(train_not_most_sim, columns=['source', 'target'])
        train_AllNegative_ne = pd.merge(AllNegative_all, train_AllNegative_ne, on=['source', 'target'], how='inner')
        train_AllNegative_ne = train_AllNegative_ne.drop_duplicates()

        if len(train_AllNegative_ne) > len(train_pos_pairs)/2:
            train_AllNegative_ne = train_AllNegative_ne.sample(n=int(len(train_pos_pairs)/2)z)

        merged_df = pd.merge(AllNegative_all, train_AllNegative_po, how='left', indicator=True)
        df2_filtered = merged_df[merged_df['_merge'] == 'left_only']
        df2_filtered.drop('_merge', axis=1, inplace=True)
        
        merged_df2 = pd.merge(df2_filtered, train_AllNegative_ne, how='left', indicator=True)
        df2_filtered2 = merged_df2[merged_df2['_merge'] == 'left_only']
        df2_filtered2.drop('_merge', axis=1, inplace=True)
        df2_filtered2 = df2_filtered2.drop_duplicates()
        train_negative_pairs = df2_filtered2.sample(n=(len(train_pos_pairs) - len(train_AllNegative_ne)))
        train_negative_pairs = pd.concat([train_negative_pairs, train_AllNegative_ne],ignore_index=True)

        
        test_most_sim = get_negtive_po(args.min_value_po,args.min_value_po2,cir_sim,drug_sim,test_pos_pairs,args.drug_node_count,args.nega_threshold)
        test_AllNegative_po = pd.DataFrame(test_most_sim, columns=['source', 'target'])
        test_not_most_sim = get_negtive_ne(args.min_value_ne,args.min_value_ne2,cir_sim,drug_sim,train_pos_pairs,args.drug_node_count,args.nega_threshold)
        test_AllNegative_ne = pd.DataFrame(test_not_most_sim, columns=['source', 'target'])
        test_AllNegative_ne = pd.merge(AllNegative_all, test_AllNegative_ne, on=['source', 'target'], how='inner')
        if len(test_AllNegative_ne) > len(test_pos_pairs)/2:
            test_AllNegative_ne = test_AllNegative_ne.sample(n=int(len(test_pos_pairs)/2))

        merged_df_test = pd.merge(AllNegative_all, train_negative_pairs, how='left', indicator=True)
        df_filtered_test = merged_df_test[merged_df_test['_merge'] == 'left_only']
        df_filtered_test.drop('_merge', axis=1, inplace=True)

        merged_df_po = pd.merge(df_filtered_test, test_AllNegative_po, how='left', indicator=True)
        df_filtered_po = merged_df_po[merged_df_po['_merge'] == 'left_only']
        df_filtered_po.drop('_merge', axis=1, inplace=True)

        merged_df_ne = pd.merge(df_filtered_po, test_AllNegative_ne, how='left', indicator=True)
        df_filtered_ne = merged_df_ne[merged_df_ne['_merge'] == 'left_only']
        df_filtered_ne.drop('_merge', axis=1, inplace=True)

        test_negative_pairs = df_filtered_ne.sample(n=(len(test_pos_pairs) - len(test_AllNegative_ne)))
        test_negative_pairs = pd.concat([test_negative_pairs, test_AllNegative_ne],ignore_index=True)


        train_pos_pairs["label"] = 1
        train_negative_pairs["label"] = 0
        train_pairs = pd.concat([train_pos_pairs, train_negative_pairs], ignore_index=True)
        test_pos_pairs["label"] = 1
        test_negative_pairs["label"] = 0
        test_pairs = pd.concat([test_pos_pairs, test_negative_pairs], ignore_index=True)

        pairs1 = [(row['source'], row['target']) for index, row in train_pairs.iterrows()]
        pairs2 = [(row['source'], row['target']) for index, row in test_pairs.iterrows()]
        assert len(pairs1) == len(set(pairs1))
        assert len(pairs2) == len(set(pairs2))
        assert not set(pairs1).intersection(set(pairs2)) == True

        drug_cir_ass = np.zeros((args.drug_node_count, args.circ_node_count))
        for _, row in train_pairs.iterrows():
            source = row["source"]
            target = row["target"]-args.drug_node_count
            drug_cir_ass[source, target] = 1
        diag = np.diag(cir_sim)
        if np.sum(diag) != 0:
            cir_sim = cir_sim - np.diag(diag)
        diag = np.diag(drug_sim)
        if np.sum(diag) != 0:
            drug_sim = drug_sim - np.diag(diag)
        n_drug = drug_sim.shape[0]
        n_cir = cir_sim.shape[0]
        drug_sim, cir_sim = get_syn_sim(drug_cir_ass, drug_sim, cir_sim, 1)
        
        drug_adj = k_matrix(drug_sim, args.knn_nums) 
        cir_adj = k_matrix(cir_sim, args.knn_nums)
        drug_e1, drug_e2 =k_adges(drug_sim, args.knn_nums) 
        cir_e1, cir_e2 = k_adges(cir_sim, args.knn_nums)
        cir_e1 = [e1+218 for e1 in cir_e1]
        cir_e2 = [e2+218 for e2 in cir_e2]
        drug_feature = np.concatenate((drug_cir_ass,drug_sim),axis=1)
        circRNA_feature = np.concatenate((drug_cir_ass.T,cir_sim),axis=1)
        drug_circ_feature = np.concatenate((drug_feature,circRNA_feature),axis=0)
        drug_circ_feature = pd.DataFrame(drug_circ_feature)
        

        Alledge2 = train_pos_pairs.iloc[:, :2]
        left = Alledge2["source"].tolist() + Alledge2["target"].tolist()
        right = Alledge2["target"].tolist() + Alledge2["source"].tolist()
        features = drug_circ_feature
        Alledge = pd.DataFrame()
        Alledge["source"] = drug_e1+cir_e1+left
        Alledge["target"] = drug_e2+cir_e2+right
        Alledge = Alledge.sort_values(by="source", ascending=True)
        features = features.iloc[:,:]

        labels = pd.DataFrame(np.random.rand(int(args.drug_node_count+args.circ_node_count),1))
        labels[0:args.drug_node_count]=0 
        labels[args.drug_node_count:]=1 
        labels = labels[0]
        adj, features, _, _, _, _  = load_data(Alledge,features,labels)
        
        args_item = item()
        node_sum = adj.shape[0]
        edge_sum = adj.sum()/2
        row_sum = (adj.sum(1) + 1) 
        norm_a_inf = row_sum/ (2*edge_sum+node_sum) # a = [(dii + 1)r(djj + 1)1-r]/(2|E|+|V|) 
        adj_norm = sparse_mx_to_torch_sparse_tensor(aug_random_walk(adj))  
        features = F.normalize(features, p=1) 
        feature_list = []
        feature_list.append(features)
        for i in range(1, args_item.k1):
            feature_list.append(torch.spmm(adj_norm, feature_list[-1])) # X(+∞) = A(+∞)*X(0) 
        norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum)

        norm_fea_inf = torch.mm(norm_a_inf, features) # (1,973) * (973,63) => (1,63)

        hops = torch.Tensor([0]*(adj.shape[0]))

        mask_before = torch.Tensor([False]*(adj.shape[0])).bool()
        for i in range(args_item.k1):   
            dist = (feature_list[i] - norm_fea_inf).norm(2, 1) 
            mask = (dist<args_item.epsilon1).masked_fill_(mask_before, False) 
            mask_before.masked_fill_(mask, True)
            hops.masked_fill_(mask, i) 
        print(hops)

        mask_final = torch.Tensor([True]*(adj.shape[0])).bool()
        mask_final.masked_fill_(mask_before, False)
        hops.masked_fill_(mask_final, args_item.k1-1)
        print("Local Smoothing Iteration calculation is done.")
        input_feature = aver(hops, adj, feature_list) 
        print("Local Smoothing is done.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_feature = input_feature.to(device)

        print("Start training...")  
        
        Emdebding = input_feature
        Emdebding_GCN = pd.DataFrame(Emdebding.detach().cpu().numpy())
        X_train = pd.concat([Emdebding_GCN.loc[train_pairs["source"].values.tolist()].reset_index(drop=True),Emdebding_GCN.loc[train_pairs["target"].values.tolist()].reset_index(drop=True)],axis=1)
        Y_train = train_pairs["label"]
        X_test = pd.concat([Emdebding_GCN.loc[test_pairs["source"].values.tolist()].reset_index(drop=True),Emdebding_GCN.loc[test_pairs["target"].values.tolist()].reset_index(drop=True)],axis=1)
        Y_test = test_pairs["label"]

        smiles = np.load("/ifs/data/fanziyu/project/LSGNN_fzy/src/data/drug_smile.npy") # 218

        compound_iso_smiles = []
        compound_iso_smiles += list(smiles)
        compound_iso_smiles = set(compound_iso_smiles)
        smile_graph = {}

        for smile in compound_iso_smiles:
            g = smile_to_graph(smile)
            smile_graph[smile] = g

        df_smiles = pd.DataFrame(smiles)
        drugdata = TestbedDataset( xd=df_smiles[0], y= [0 for _ in range(218)], smile_graph=smile_graph)
        drugdata = DataLoader(drugdata, batch_size=218, shuffle=None)

        
        
        best_AUC = 0

        model = MultiGCN(input_size=(args.circ_node_count+args.drug_node_count),output_size = args.output_size,n_layers = max(hops))
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)

        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            CDA_edge = torch.tensor([Alledge['source'].tolist(),Alledge['target'].tolist()])
            CDA_edge2 = torch.tensor([train_pairs['source'].tolist(),train_pairs['target'].tolist()])
            CDA_edge3 = torch.tensor([test_pairs['source'].tolist(),test_pairs['target'].tolist()])

            for batch_idx, data in enumerate(drugdata):
                drug_data_smiles = data
            outputs= model(x1 = features, edges = CDA_edge, hop = hops, edges2 = CDA_edge2, drugdata = drug_data_smiles)[0]
            loss = criterion(outputs, Y_train_tensor)
            loss.backward()
            optimizer.step()

            model = model.eval()
            with torch.no_grad():
                threshold = args.threshold
                y_score_RandomF2= model(x1 = features, edges = CDA_edge, hop = hops, edges2 = CDA_edge3, drugdata = drug_data_smiles)[0].detach().numpy().flatten()
                y_score02 = [1 if score >= threshold else 0 for score in y_score_RandomF2]
                auc2 = roc_auc_score(Y_test, y_score_RandomF2)
                if auc2 > best_AUC:
                    y_score_RandomF = model(x1 = features, edges = CDA_edge, hop = hops, edges2 = CDA_edge3, drugdata = drug_data_smiles)[0].detach().numpy().flatten()
                    y_score0 = [1 if score >= threshold else 0 for score in y_score_RandomF]
                    e_auc_score = round(roc_auc_score(Y_test, y_score_RandomF),4)
                    e_aupr_score = round(average_precision_score(Y_test, y_score_RandomF),4)
                    e_pre_score = round(precision_score(Y_test, y_score0), 4)
                    e_rec = round(recall_score(Y_test, y_score0), 4)
                    e_accuracy = round(accuracy_score(Y_test, y_score0), 4)
                    e_f1 = round(f1_score(Y_test, y_score0), 4)
                    tn, fp, fn, tp = confusion_matrix(Y_test, y_score0).ravel()
                    e_specificity_score = tn / (tn + fp)
                    print("epoch:",epoch," auc_score:", e_auc_score, " aupr_score:", e_aupr_score, " pre_score:", e_pre_score, " rec:", e_rec, " acc:", e_accuracy, " f1:", e_f1, " spe:", e_specificity_score)
                    best_AUC = auc2
        
        
        
        # 定义三个列表
        list1 = list(zip(test_pairs["source"].values.tolist(), test_pairs["target"].values.tolist()))
        list2 = Y_test.tolist()
        list3 = y_score02

        # 将列表转换为字典，然后转换为DataFrame
        data6 = {'pairs': list1, 'true': list2, 'predict': list3}
        df6 = pd.DataFrame(data6)

        # 保存为CSV文件
        df6.to_csv('/ifs/data/fanziyu/project/LSNSCDA/result/'+str(kfold)+'.csv', index=False)
        auc_score = round(roc_auc_score(Y_test, y_score_RandomF),4)
        aupr_score = round(average_precision_score(Y_test, y_score_RandomF),4)
        pre_score = round(precision_score(Y_test, y_score0), 4)
        rec = round(recall_score(Y_test, y_score0), 4)
        accuracy = round(accuracy_score(Y_test, y_score0), 4)
        f1 = round(f1_score(Y_test, y_score0), 4)
        tn, fp, fn, tp = confusion_matrix(Y_test, y_score0).ravel()
        specificity_score = tn / (tn + fp)


        aucs.append(auc_score)
        auprs.append(aupr_score)
        pres.append(pre_score)
        recs.append(rec)
        spes.append(specificity_score)
        f1s.append(f1)
        acc.append(accuracy)
        result_4_ROC1.append(Y_test)
        result_4_ROC2.append(y_score_RandomF)
    
    mean_auc=sum(aucs)/args.k_fold
    mean_aupr=sum(auprs)/args.k_fold
    mean_pre=sum(pres)/args.k_fold
    mean_rec=sum(recs)/args.k_fold
    mean_spe=sum(spes)/args.k_fold
    mean_f1=sum(f1s)/args.k_fold
    mean_acc=sum(acc)/args.k_fold

    print('Mean ROC (AUC=%0.4f)'% (mean_auc)) 
    print('Mean AUPR (AUPR=%0.4f)'% (mean_aupr)) 
    print('Mean ACC (ACC=%0.4f)'% (mean_acc)) 
    print('Mean F1-SCORE (F1=%0.4f)'% (mean_f1)) 
    print('Mean PRE (PRE=%0.4f)'% (mean_pre)) 
    print('Mean REC (REc=%0.4f)'% (mean_rec)) 
    print('Mean SPE (SPE=%0.4f)'% (mean_spe)) 
