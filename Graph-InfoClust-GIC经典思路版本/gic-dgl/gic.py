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


class GIC(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, K, beta, alpha):
        super(GIC, self).__init__()
        self.n_hidden = n_hidden
        self.g=g  # 输入的图.
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        # self.discriminator2 = Discriminator(n_hidden)
        self.discriminatorK = DiscriminatorK(n_hidden)
        self.loss = nn.BCEWithLogitsLoss() # binary cross entropy # 设计一个更复杂的损失函数. 参考.gan网络, 给他补一个生成器.  Graph gan---------graph embedding -----生物-------玄学-------非解释性.
        self.K = K
        self.beta = beta
        self.cluster = Clusterator(n_hidden,K)
        self.alpha = alpha
        

    def forward(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
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
        
        
        l += (1-self.alpha)*(self.loss(pos_cluster, torch.ones_like(pos_cluster)) + self.loss(neg_cluster, torch.zeros_like(neg_cluster))) 
                          
        
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