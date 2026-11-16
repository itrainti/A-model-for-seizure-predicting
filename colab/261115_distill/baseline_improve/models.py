import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

out_channals = 32
embed_dim = 16
num_heads = 2
hidden_size = 20


class ConvPaddingSame(nn.Module):
    def __init__(self):
        super().__init__()

    def Conv1d(self,in_c,out_c,kernel_size,stride=1,groups=1,bias=False):
        padding = kernel_size // 2
        return nn.Conv1d(in_c,out_c,kernel_size,stride=stride,padding=padding,groups=groups,bias=bias)

    def Conv2d(self,in_c,out_c,kernel_size,stride=1,groups=1,bias=False):
        padding = kernel_size // 2
        return nn.Conv2d(in_c,out_c,kernel_size,stride=stride,padding=padding,bias=bias)

    def ConvTranspose1d(self,in_c,out_c,kernel_size,stride=1,groups=1,bias=False):
        padding = kernel_size // 2
        output_padding = stride - 1
        return nn.ConvTranspose1d(in_c,out_c,kernel_size,stride=stride,padding=padding,groups=groups,bias=bias,output_padding=output_padding)


class AutoEncoder(nn.Module):
    def __init__(self,in_channals=1,out_channals=out_channals,kernel_size=3,n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.in_channals = in_channals
        self.out_channals = out_channals
        self.cps = ConvPaddingSame()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def depthwise_sep_conv1d(self,in_channals,out_channals,kernel_size):
        return nn.Sequential(
            # 深度卷积变长度
            self.cps.Conv1d(
                in_channals,in_channals,kernel_size,
                stride=2,bias=True,groups=in_channals
            ),
            nn.BatchNorm1d(in_channals),
            nn.ReLU(),
            # 逐点卷积变通道
            self.cps.Conv1d(in_channals,out_channals,kernel_size=1,stride=1),
            nn.BatchNorm1d(out_channals),
            nn.ReLU(),
        )

    # 编码器
    def build_encoder(self):
        n = max(self.n_layers-2,0)
        layers = [self.depthwise_sep_conv1d(self.out_channals,self.out_channals,self.kernel_size) for _ in range(n)]
        return nn.Sequential(
            # 1 -> out_channals
            self.depthwise_sep_conv1d(self.in_channals,self.out_channals,self.kernel_size),
            *layers,
            # 取平均值
            nn.AvgPool1d(self.kernel_size,stride=2,padding=self.kernel_size//2)
        )

    # 解码器
    def build_decoder(self):
        # output_padding = stride + 2padding - kernel_size
        layers = [self.cps.ConvTranspose1d(self.out_channals,self.out_channals,self.kernel_size,stride=2) for _ in range(self.n_layers-1)]
        return nn.Sequential(
            *layers,
            self.cps.ConvTranspose1d(self.out_channals,self.in_channals,self.kernel_size,stride=2)
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CommonConv1d(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cps = ConvPaddingSame()
        self.cl = [1,2,4,8,16,32]
        self.model = self.build_model()

    def cell_layer(self,in_c,out_c):
        return nn.Sequential(
            self.cps.Conv1d(in_c,out_c,kernel_size=3,stride=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )

    # channals = 1 -> [2,4,8,16,32]
    def build_model(self,):
        m = [self.cell_layer(self.cl[i-1],self.cl[i]) for i in range(1,len(self.cl))]
        return nn.Sequential(*m)

    def forward(self,x):
        return self.model(x)


class CommonConv2d(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cl = [1,2,4,8,16,32]
        self.model = self.build_model()

    def cell_layer(self,in_c,out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=(1,3),stride=1,padding="same"),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
        )

    # channals = 1 -> [2,4,8,16,32]
    def build_model(self):
        m = [self.cell_layer(self.cl[i-1],self.cl[i]) for i in range(1,len(self.cl))]
        return nn.Sequential(*m)

    def forward(self,x): # (B,1,6,L)
        return self.model(x) # (B,32,6,L/i) i=2^len(self.c1)


class TemporalAttention(nn.Module):
    def __init__(self,channals):
        super().__init__()
        self.fc = nn.Linear(channals,1)

    def attention(self,x):
        scores = self.fc(x) # (N,L,C) -> (N,L,1)
        weights = F.softmax(scores,dim=1) # (N,L,1)
        atten = weights * x # (N,L,C)
        return weights,atten

    def forward(self,x):
        return self.attention(x)


class ChannalAttention(nn.Module):
    def __init__(self,channals):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channals,channals // 2),
            nn.LayerNorm(channals // 2),
            nn.ReLU(),
            nn.Linear(channals // 2,channals),
            nn.Sigmoid()
        )

    def attention(self,x):
        pooled = x.mean(dim=1) # (N,C) 压缩时间序列但原序列不变
        weights = self.fc(pooled).unsqueeze(dim=1) # (N,1,C)
        atten = weights * x # (N,L,C)
        return weights,atten

    def forward(self,x):
        return self.attention(x)


class ElecAfterMultiHeadAttention(nn.Module):
    def __init__(self,elec_nums=6,embed_dim=embed_dim):
        super().__init__()
        # 必须满足 embed_dim % num_heads == 0
        self.elec_multi_atten = nn.MultiheadAttention(embed_dim=embed_dim,batch_first=True,num_heads=num_heads)
        # 广播之后，同一电极每个概率共享一个特殊编码
        self.emb = nn.Parameter(torch.randn(1,elec_nums,embed_dim)*0.1)
        self.projecter = nn.Linear(1,embed_dim) # probs = (W,1)
        self.reducer = nn.Linear(embed_dim+1,1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x): # (N,n_E,W)
        B,E,W = x.shape
        h = x.permute(0,2,1).unsqueeze(-1).reshape(B*W,E,-1) # (N*W,n_E,1)
        h = self.projecter(h) + self.emb # (N*W,n_E,embed_dim)
        a,_ = self.elec_multi_atten(h,h,h)
        a = self.norm(a+h) # 残差连接并归一化
        # 连接上原概率，防止盲猜
        a = torch.cat([a, x.permute(0,2,1).reshape(B*W,E,1)], dim=-1)
        scores = self.reducer(a).squeeze(-1)
        weights = F.softmax(scores,dim=-1) # (N*W,n_E)
        x = x.permute(0,2,1).reshape(B*W,E)
        x = (weights * x).sum(dim=-1,keepdim=True)
        return x


class BILSTMClassify(nn.Module):
    def __init__(self,input_size=out_channals,hidden_size=10,num_layers=1,dropout=0,temporal_attention=False,channal_attention=False): # input_size是encoder输出的channals
        super().__init__()
        self.BI_LSTM = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size, # 隐藏状态维度
            num_layers = num_layers, # 隐藏层层数
            batch_first = True,
            bidirectional = True,  # 开启双向
        )
        self.fc = nn.Linear(2*hidden_size,1)
        self.dropout = nn.Dropout(dropout)
        self.temporal_attention = temporal_attention
        self.channal_attention = channal_attention
        # 时序注意力
        if self.temporal_attention:
            self.temp_atten = TemporalAttention(2*hidden_size)
        # 通道注意力
        if self.channal_attention:
            self.chan_atten = ChannalAttention(2*hidden_size)

    def forward(self,x):
        x,_ = self.BI_LSTM(x)
        if self.channal_attention:
            _,x = self.chan_atten(x)
        if self.temporal_attention:
            _,x = self.temp_atten(x)
            out = x.sum(dim=1)
        else:
            out = x[:,-1,:]
        out = self.dropout(out)
        return self.fc(out)


class EncoderBILSTMClassify(nn.Module):
    def __init__(self,weight_path,in_channals=1,out_channals=out_channals,kernel_size=3,n_layers=3,input_size=out_channals,hidden_size=10,num_layers=1,dropout=0,temporal_attention=False,channal_attention=False):
        super().__init__()
        self.ae = AutoEncoder(in_channals,out_channals,kernel_size,n_layers)
        self.ae.load_state_dict(torch.load(weight_path)) # 加载权重
        self.encoder = self.ae.build_encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False # 冻结权重
        self.classify = BILSTMClassify(input_size,hidden_size,num_layers,dropout,temporal_attention,channal_attention)

    def forward(self,x):
        x = self.encoder(x)
        x = x.permute(0,2,1)
        return self.classify(x)


class ConvBILSTMClassify(nn.Module):
    def __init__(self,input_size=out_channals,hidden_size=hidden_size,num_layers=1,dropout=0,temporal_attention=False,channal_attention=False):
        super().__init__()
        self.common_conv = CommonConv1d()
        self.classify = BILSTMClassify(out_channals,hidden_size,num_layers,dropout,temporal_attention,channal_attention)

    def forward(self,x):
        x = self.common_conv(x)
        x = x.permute(0,2,1)
        return self.classify(x)


ELECS = ["Fp1-F7", "Fp2-F8", "T7-P7", "T8-P8", "P7-O1", "P8-O2"]


def get_positions(elec_names, montage_name="standard_1020"):
    montage = mne.channels.make_standard_montage(montage_name)
    ch_list, dig_list = montage.ch_names, montage.dig
    pos = []
    for name in elec_names:
        a, b = name.split("-")
        # standard_1020 里是 'Fp1'、'F7'，注意大小写
        va = np.array(dig_list[ch_list.index(a)]['r'])
        vb = np.array(dig_list[ch_list.index(b)]['r'])
        pos.append((va + vb) / 2.0)
    return torch.tensor(np.stack(pos), dtype=torch.float32)  # (6,3)


def build_adjacency(elec_names):
    """距离倒数邻接矩阵，对角线为 0"""
    p = get_positions(elec_names)
    n = len(elec_names)
    adj = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = torch.norm(p[i] - p[j])
                if d > 0:
                    adj[i, j] = 1.0 / d
    return adj  # (6, 6)


class GraphConstructor(nn.Module):
    def __init__(self, num_nodes=6, features=32, hidden_dim=128, pos_adj=build_adjacency(ELECS)):
        # features 是输入的特征维度
        # hidden_dim 是动态图计算的隐层维度
        super().__init__()
        self.num_nodes = num_nodes
        # 全局静态图
        self.static_weights = nn.Parameter(torch.eye(num_nodes))
        # 沿最后一维 (B,L,N,N) 的 j 维做归一化，dim=3
        self.similarity_net = nn.Sequential(
            nn.Linear(features*2, hidden_dim),
            nn.ELU(inplace=False),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=3))
        # 空间图权重
        self.pos_factor = nn.Parameter(torch.tensor(0.5))
        # 邻接矩阵和x的维度要对上，保证能做矩阵乘法
        self.register_buffer("pos_encoding", pos_adj.clone())

    # 局部的动态图
    def compute_similarity(self, x):  # x: (B, C, 6, L/32)
        B, D, N, L = x.shape
        x = x.reshape(B, L, N, D)
        node_i = x.unsqueeze(2).expand(-1, -1, N, -1, -1)
        node_j = x.unsqueeze(3).expand(-1, -1, -1, N, -1)
        # 两个维度拼接
        pairs = torch.cat([node_i, node_j], dim=-1)  # (B, L, N, N, 2D)
        return self.similarity_net(pairs).squeeze(-1)  # (B, L, N, N)

    def forward(self, x):  # x: (B, C, 6, L/32)
        static_adj = F.softmax(self.static_weights, dim=1)
        dynamic_adj = self.compute_similarity(x)
        pos_adj = F.softmax(self.pos_encoding, dim=1)
        return F.softmax(static_adj + dynamic_adj + self.pos_factor * pos_adj, dim=3)  # (B,L,N,N)


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=128):
        """
        in_channels: 输入节点特征维度
        out_channels: 输出节点特征维度
        hidden_dim: 动态图计算隐层维度
        """
        super().__init__()
        # 拼接要按输入特征维度，所以 features 必须跟着 in_channels 走
        self.gc = GraphConstructor(features=in_channels, hidden_dim=hidden_dim)
        # 对每个节点的特征做线性变换
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):  # x: (B, C, N, L)
        B, C, N, L = x.shape
        # 1. 图卷积
        adjacency = self.gc(x)  # (B, L, N, N)
        # 2. 调整节点特征维度
        x = x.permute(0, 3, 2, 1)  # (B, L, N, C)
        # 3. 合并样本维度和时间维度
        adjacency = adjacency.reshape(B * L, N, N)
        x = x.reshape(B * L, N, C)
        # 4. 空间聚合
        x = torch.bmm(adjacency, x)  # (B*L, N, C)
        # 5. 特征变换
        x = self.fc(x)  # (B*L, N, out_channels)
        # 6. 恢复维度
        x = x.reshape(B, L, N, -1)
        x = x.permute(0, 3, 2, 1)
        return x  # (B, out_channels, N, L)


class Detection(nn.Module):
    def __init__(self, in_channals=1, out_channals=32, embed_dim=16, num_heads=2,
                 temporal_attention=False, channal_attention=False, dropout=0):
        super().__init__()
        # 输入是 CommonConv2d 的输出 (B,32,6,L/32)，所以第一层 in_channels=32
        self.gcn = nn.Sequential(
            GraphConvolution(in_channels=32, out_channels=16, hidden_dim=64),
            GraphConvolution(in_channels=16, out_channels=32, hidden_dim=128)
        )
        self.conv2d = CommonConv2d()
        self.Bilstm = BILSTMClassify(
            temporal_attention=temporal_attention,
            channal_attention=channal_attention,
            dropout=dropout
        )
        # 图卷积和卷积融合
        self.fc = nn.Linear(out_channals*2, out_channals)

    def forward(self, x):  # (B,6,L)
        x1 = x.unsqueeze(1)  # (B,1,6,L)
        # 卷积分支
        conv_feat = self.conv2d(x1)  # (B, 32, 6, L/32)
        # 图卷积分支
        gcn_feat = self.gcn(conv_feat)  # (B, 32, 6, L/32)
        x = torch.cat((gcn_feat, conv_feat), dim=1)  # (B, 32*2, 6, L/32)
        x = x.permute(0, 2, 3, 1)  # (B, 6, L/32, 64)
        x = self.fc(x)  # (B, 6, L/32, 32)

        # 时序预测，每个电极一个 logit，再对六通道取平均；已经是 (B,6,L,C)，直接展平
        B, N, L, C = x.shape
        x = x.reshape(B*N, L, C)
        x = self.Bilstm(x)  # (B*6, 1)
        return x.reshape(B, N).mean(dim=1, keepdim=True)  # (B, 1)
