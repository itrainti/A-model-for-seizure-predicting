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
    def __init__(self):
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
