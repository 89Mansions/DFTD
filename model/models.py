import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from einops import rearrange


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class FAttn(nn.Module):
    def __init__(self, in_dim):
        super(FAttn, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.global_maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1_1 = nn.Linear(in_dim, in_dim//2)
        self.fc2 = nn.Linear(in_dim//2, in_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.global_avgpool(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1_1(x1)
        x2 = self.global_maxpool(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1_1(x2)
        temp = x1 + x2
        temp_a = self.fc2(temp)
        alpha = self.softmax(temp_a)
        feature = (alpha.unsqueeze(-1)) * x
        return feature
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0):
        super(TemporalBlock, self).__init__()
        kernel_size1 = kernel_size
        padding = dilation * (kernel_size1-1)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=kernel_size1,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=kernel_size1,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net1 = nn.Sequential(self.conv1, self.relu1, self.chomp1, self.dropout1)
        self.net2 = nn.Sequential(self.conv2, self.relu2, self.chomp2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.fattn = FAttn(in_dim=n_outputs)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x2 = self.fattn(x2)
        out = x2
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out+res)

class chan_selfattn(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(chan_selfattn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) n -> b head c (n)', head=self.num_heads)  # b*C*n
        k = rearrange(k, 'b (head c) n -> b head c (n)', head=self.num_heads)  # b*C*n
        v = rearrange(v, 'b (head c) n -> b head c (n)', head=self.num_heads)  # b*C*n
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # heads*C*C
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (n) -> b (head c) n', head=self.num_heads, n=n)
        out = self.project_out(out)
        return out
    
class seq_selfattn(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(seq_selfattn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) n -> b head c (n)', head=self.num_heads)  # b*C*n
        k = rearrange(k, 'b (head c) n -> b head c (n)', head=self.num_heads)  # b*C*n
        v = rearrange(v, 'b (head c) n -> b head c (n)', head=self.num_heads)  # b*C*n
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=1)
        attn = (k.transpose(-2, -1) @ q) * self.temperature  # heads*n*n
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2, -1))
        out = rearrange(out, 'b head (n) c -> b (head c) n', head=self.num_heads, n=n)
        out = self.project_out(out)
        return out
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, tcn_OutputChannelList, dropout=0):
        super(TemporalConvNet, self).__init__()
        self.block1 = TemporalBlock(n_inputs = num_inputs, n_outputs = tcn_OutputChannelList[0], kernel_size=2 , stride=1, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(n_inputs = tcn_OutputChannelList[0], n_outputs = tcn_OutputChannelList[1], kernel_size=2, stride=1, dilation=2, dropout=dropout)
        self.block3 = TemporalBlock(n_inputs = tcn_OutputChannelList[1], n_outputs = tcn_OutputChannelList[2], kernel_size=2, stride=1, dilation=4, dropout=dropout)
        self.block1_1 = TemporalBlock(n_inputs = num_inputs, n_outputs = tcn_OutputChannelList[0], kernel_size=3 , stride=1, dilation=1, dropout=dropout)
        self.block2_1 = TemporalBlock(n_inputs = tcn_OutputChannelList[0], n_outputs = tcn_OutputChannelList[1], kernel_size=3, stride=1, dilation=1, dropout=dropout)
        self.block3_1 = TemporalBlock(n_inputs = tcn_OutputChannelList[1], n_outputs = tcn_OutputChannelList[2], kernel_size=3, stride=1, dilation=1, dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=tcn_OutputChannelList[2]*2,out_channels=tcn_OutputChannelList[2],kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=tcn_OutputChannelList[2]*2,out_channels=tcn_OutputChannelList[2],kernel_size=1)
        # self.conv1 = nn.Sequential(weight_norm(nn.Conv1d(in_channels=tcn_OutputChannelList[2]*2,out_channels=tcn_OutputChannelList[2],kernel_size=1)),nn.ReLU())
        # self.conv2 = nn.Sequential(weight_norm(nn.Conv1d(in_channels=tcn_OutputChannelList[2]*2,out_channels=tcn_OutputChannelList[2],kernel_size=1)),nn.ReLU())

        self.chan_attn = chan_selfattn(dim=tcn_OutputChannelList[2], num_heads=4)
        self.seq_attn = seq_selfattn(dim=tcn_OutputChannelList[2], num_heads=4)

    def forward(self, x): 
        """
        
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        x1_1 = self.block1_1(x)
        x2_1 = self.block2_1(x1_1)
        x3_1 = self.block3_1(x2_1)
        locglo_fusion = self.conv1(torch.cat([x3, x3_1], dim=1))
        chan_attn = self.chan_attn(locglo_fusion)
        seq_attn = self.seq_attn(locglo_fusion)
        out = self.conv2(torch.cat([chan_attn, seq_attn], dim=1)) + locglo_fusion
        return out


class Net(nn.Module):
    def __init__(self, input_size, tcn_OutputChannelList, dropout):
        super(Net, self).__init__()
        self.tcn = TemporalConvNet(input_size, tcn_OutputChannelList, dropout=dropout)

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs = inputs.transpose(1, 2)
        y = self.tcn(inputs)
        return y


class Model(nn.Module):
    def __init__(self, input_features_num, input_len, tcn_OutputChannelList, output_len,  tcn_Dropout):
        super(Model, self).__init__()
        self.tcnunit = Net(input_features_num, tcn_OutputChannelList, tcn_Dropout)
        self.linear = nn.Linear(tcn_OutputChannelList[-1]*input_len, output_len)

    def forward(self, input_seq):
        tcn_out = self.tcnunit(input_seq)
        # print(tcn_out.shape)
        tcn_out = tcn_out.view(tcn_out.size(0), -1)
        out = self.linear(tcn_out)
        return out

if __name__=="__main__":
    x = torch.randn(1, 48, 16)
    model = Model(input_features_num=16, input_len=48, tcn_OutputChannelList=[32, 64, 128], output_len=24, tcn_Dropout= 0.1)
    y = model(x)
    print(y.shape)
