"""
Graph InfoClust in DGL
Implementation is based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from gcn import GCN

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt: # corrupt 破坏数据, 也就是创造假的数据,通过一个随机置换.
            perm = torch.randperm(self.g.number_of_nodes())  #造假,感觉还不够假.
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        #features @ self.weight @ summary.t()  # 只是一个特征变换, 保证输出shape对上就行.
        return torch.matmul(features, torch.matmul(self.weight, summary))
    
class DiscriminatorK(nn.Module):
    def __init__(self, n_hidden):
        super(DiscriminatorK, self).__init__()

    def forward(self, features, summary):
        
        n, h = features.size()
        
        ####features =  features / features.norm(dim=1)[:, None]
        #features = torch.sum(features*summary, dim=1)
        
        #features = features @ self.weight @ summary.t()
        return torch.bmm(features.view(n, 1, h), summary.view(n, h, 1)) #torch.sum(features*summary, dim=1) 



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
#使用时候h输入features.  2078*1433
    def forward(self, h, adj):  # h 是输入的特征. 我们先乘以变换矩阵, 改变他的dimension. 思路就是因为之前的维度1433太大了.我们计算量太大, 所以改变为一个小的维度便于计算.
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(
            Wh)  # a_input 表示的含义,  a是 2708* 2708*16 的2708维矩阵  所以一个元素 i,j 表示 i的8维特征和j 的8维特征concat成16维. # 这个步奏特别耗内存!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)  # 负无穷
        attention = torch.where(adj > 0, e,
                                zero_vec)  # 小于=0的都变成负无穷. adj临街矩阵大于0的都算e. 不要注意力为负数的部分.  这个部分感觉可以省略试试.感觉作用不大.毕竟负数softmax也很小.
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)  # elu是一个激活函数.
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 每一行重复 2708次.
        Wh_repeated_alternating = Wh.repeat(N, 1)  # 这个是每一次重复2708行.重复2708次.
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,corrupt):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj,corrupt=False):
        if corrupt: # corrupt 破坏数据, 也就是创造假的数据,通过一个随机置换.
            perm = torch.randperm((len(x)))  #造假,感觉还不够假.  得到0 到len(x)-1的一个置换.
            x = x[perm]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)) # 再做一层att, 再加一个elu
        return x  # 注意nll的使用方法:

























class GIC(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, K, beta, alpha,gamma):
        super(GIC, self).__init__()
        self.n_hidden = n_hidden
        self.g=g  # 输入的图.
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        # self.discriminator2 = Discriminator(n_hidden)
        self.discriminatorK = DiscriminatorK(n_hidden)
        #我们2021-04-05,11点58 搭建一个新网络graph attentin网络来加入一个新特征.
        self.gamma=gamma
        self.gat=GAT(nfeat=n_hidden,   # 这里我们走这个代码, 不走稀疏矩阵.
                nhid=8, #这个要去特别小.  跟gcn没法比.
                nclass=2,  #这里面只提取特征. -----直接判断真假.
                dropout=0.1,
                nheads=1,  # 8个头太卡了.
                alpha=0.2
                     ,corrupt=False)




        self.loss = nn.BCEWithLogitsLoss() # binary cross entropy # 设计一个更复杂的损失函数. 参考.gan网络, 给他补一个生成器.  Graph gan---------graph embedding -----生物-------玄学-------非解释性.
        self.K = K
        self.beta = beta
        self.cluster = Clusterator(n_hidden,K)
        self.alpha = alpha
        

    def forward(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)

        adj=self.g.adj() # 稀疏矩阵查看方法:adj.to_dense().numpy()
        adj=adj.to_dense()
        attention_pos = self.gat(positive, adj,corrupt=False) # 直接输出真假概率.
        attention_neg = self.gat(negative, adj,corrupt=False)
        graph_summary = torch.sigmoid(positive.mean(dim=0))# summary平均数用来聚类.
        
        mu, r = self.cluster(positive, self.beta) # r是每一个node距离mu的相似度.relative.
        
        
        cluster_summary = torch.sigmoid(r @ mu)  # 聚类特征. 创新点.  cluster_summary 含义是带有聚类特征的各个点特征.
        
        pos_graph = self.discriminator(positive, graph_summary)  # 可以修改,加入更多特征.
        neg_graph = self.discriminator(negative, graph_summary)
        

        l1 = self.loss(pos_graph, torch.ones_like(pos_graph))   # 取1 ,算loss, 这个是分类的loss
        l2 = self.loss(neg_graph, torch.zeros_like(neg_graph)) 

        l = self.alpha*(l1+l2)
        
        
        pos_cluster = self.discriminatorK(positive, cluster_summary)  # 下面算 聚类的loss. 这行算出pos的概率
        neg_cluster = self.discriminatorK(negative, cluster_summary)  # nlp :上下文.
        
        
        l += (1-self.alpha)*(self.loss(pos_cluster, torch.ones_like(pos_cluster)) + self.loss(neg_cluster, torch.zeros_like(neg_cluster))) +       self.gamma*(self.loss(attention_pos, torch.ones_like(attention_pos)) + self.loss(attention_neg, torch.zeros_like(attention_neg)))


        # l +=     self.gamma*(self.loss(attention_pos, torch.ones_like(attention_pos)) + self.loss(attention_neg, torch.zeros_like(attention_neg)))

        # l += (1-self.alpha)*(self.loss(pos_cluster, torch.ones_like(pos_cluster)) + self.loss(neg_cluster, torch.zeros_like(neg_cluster)))
                          
        
        return l

def cluster(data, k, temp, num_iter, init, cluster_temp):
    '''  # kmean算法. k是128. init是128*64的随机矩阵,
    pytorch (differentiable) implementation of soft k-means clustering. 
    Modified from https://github.com/bwilder0/clusternet
    '''
    cuda0 = torch.cuda.is_available()#False  # 把总共data的2486*64的点进行聚类
    
    
    
    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init  # init 先随机找到128个点.
    n = data.shape[0]
    d = data.shape[1]

    data = data / (data.norm(dim=1) + 1e-8)[:, None]  # 首先进行数据归一化.每一个node特征除以自己的norm
    
    for t in range(num_iter):
        #get distances between all data points and cluster centers
        
        mu = mu / mu.norm(dim=1)[:, None]  # mu是上来随机找到的 128个点. 作为聚点.
        dist = torch.mm(data, mu.transpose(0,1)) # 2个模唱为1的向量,内机越大,表示距离越近. 所以左边其实表示的是-dist
        
        
        #cluster responsibilities via softmax
        r = F.softmax(cluster_temp*dist, dim=1)  # r 代表的含义是 dist 距离当前128个聚类点的比例
        #total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        #mean of points in each cluster weighted by responsibility
        cluster_mean = r.t() @ data # r 就得到了更新.
        #update cluster means
        new_mu = torch.diag(1/cluster_r) @ cluster_mean  # 128*64            这个只是除以总的权重,维持归一化.
        mu = new_mu
        
    

    r = F.softmax(cluster_temp*dist, dim=1)
    
    
    return mu, r  # relation 相似度.

class Clusterator(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the 
    embeddings and the the node similarities (just output for debugging purposes).
    
    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to 
    run the k-means updates for.
    Modified from https://github.com/bwilder0/clusternet
    '''
    def __init__(self, nout, K):
        super(Clusterator, self).__init__()

        # 聚类方法叫 软kmeans
        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout
        
        self.init =  torch.rand(self.K, nout)
        
    def forward(self, embeds, cluster_temp, num_iter=10):
        # cluster_temp 软系数. 越大,越硬.越接近 标准kmean
        mu_init, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp = torch.tensor(cluster_temp), init = self.init)
        #self.init = mu_init.clone().detach()  在上面的结果上再做一次聚类.
        mu, r = cluster(embeds, self.K, 1, 1, cluster_temp = torch.tensor(cluster_temp), init = mu_init.clone().detach())
        
        return mu, r
    
    

    
class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)