import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from itertools import islice
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gmean
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='', dataset='drugtographcla',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    def process(self, xd, y, smile_graph):
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            labels = y[i]
            c_size, features, edge_index = smile_graph[smiles]
            if len(edge_index) != 0:
                GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index= torch.LongTensor(edge_index).transpose(1, 0), 
                                    y=torch.Tensor([labels]))
            else:
                GCNData = DATA.Data(x=torch.Tensor(features),
                    edge_index= torch.LongTensor(edge_index),
                    y=torch.Tensor([labels]))

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GCNNet(torch.nn.Module):
    def __init__(self, n_output=2,num_features_xd=78, dropout=0.2):

        super(GCNNet, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, 489)
        self.final = torch.nn.Linear(489 + num_features_xd*2, 489)
  

    def forward(self, data1, drug2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.dropout(self.relu(x1))
        x1 = gmp(x1, batch1)      
        x1 = self.relu(self.drug1_fc_g1(x1))

        f = x1 + drug2
        return f

class MultiGCN(nn.Module):
    def __init__(self,input_size, output_size, n_layers, dropout=0.2):
        super(MultiGCN, self).__init__()
        self.smiles_gcn = GCNNet()
        self.num_layers = n_layers
        self.dropout = dropout
        layer_stack = []

        for i in range(int(self.num_layers)):
            layer_stack.append(GCNConv(in_channels=input_size, out_channels=input_size, cached=False))
        self.layer_stack = nn.ModuleList(layer_stack)

        self.fc1 = nn.Linear(input_size, output_size)
        self.CDA = CDA_Decoder(Nodefeat_size = 2*(input_size+output_size),nhidden = [output_size,output_size,output_size], nlayers = 3)
    
    def forward(self, x1, edges,hop,edges2,drugdata):
        edges = edges.long()
        result = []
        xx = []

        Drugs = x1[:218, :]
        CircRNAs = x1[218:,:]
        Drugs = self.smiles_gcn(drugdata, Drugs)
        x1 = torch.cat((Drugs,CircRNAs),dim = 0)

        for idx, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x=x1, edge_index=edges)
            if idx < self.num_layers:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
                xx.append(x)
        num = 0
        for i in hop:
            result.append(xx[int(i)-1][num])
            num+=1
        x = torch.stack(result)
        x = F.relu(self.fc1(x))

        x = torch.cat((x, x1), 1)
        output = self.CDA(x,edges2[1],edges2[0]).squeeze()

        all_edge = []
        all_edge1 = []
        all_edge2 = []
        for i in range(218):
            for j in range(218,489):
                all_edge1.append(i)
                all_edge2.append(j)
        all_edge.append(all_edge1)
        all_edge.append(all_edge2)

        all_edge = torch.tensor(all_edge)
        output2 = self.CDA(x,all_edge[1],all_edge[0]).squeeze()
        return output, output2.view(218,271)

class CDA_Decoder(nn.Module):
    def __init__(self, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(CDA_Decoder, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.nlayers = nlayers
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else nhidden[l - 1], nhidden[l]) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l]) for l in range(nlayers)])
        self.linear = torch.nn.Linear(nhidden[nlayers-1], 1)

    def forward(self, nodes_features, circRNA_index, drug_index):
        RNA_features = nodes_features[circRNA_index]
        drug_features = nodes_features[drug_index]
        pair_nodes_features = torch.cat((RNA_features, drug_features), 1) # (6612, 2*(489 + 256))

        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = self.dropout_layer(pair_nodes_features)
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))
            pair_nodes_features = self.BatchNormList[l](pair_nodes_features)
        pair_nodes_features = self.dropout_layer(pair_nodes_features)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output)

class MultiGAT(nn.Module):
    def __init__(self, n_units=[17, 128, 100], n_heads=[2, 2], dropout=0.0):
        super(MultiGAT, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        for i in range(self.num_layers):
            in_channels = n_units[i] * n_heads[i-1] if i else n_units[i]
            layer_stack.append(GATConv(in_channels=in_channels, out_channels=n_units[i+1], cached=False, heads=n_heads[i]))
        
        self.layer_stack = nn.ModuleList(layer_stack)
    def forward(self, x, edges):
        
        for idx, gat_layer in enumerate(self.layer_stack):
            x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, edges)
            if idx+1 < self.num_layers:
                x = F.elu(x)
        
        return x

if __name__ == '__main__':
    
    hiddenUnits=[17, 128, 128]
    heads = [2, 2]
    
    numFeatures = 3
    
    x = torch.randn((19, numFeatures))
    edges = torch.Tensor([[1, 2], [2, 3], [1, 3]])
    edges = torch.transpose(edges, 0, 1).to(torch.int64)


    print(x.shape, edges.shape)
    model = MultiGCN(n_units=[3, 256, 256])
    out = model(x, edges)
    print(out.size())
