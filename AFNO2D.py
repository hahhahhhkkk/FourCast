import torch
from torch import nn
import torch.nn.functional as F


# 假设hidden_dim 256    256//8=32 每块32
class AFNO2D(nn.Module):
    def __init__(self,input_size,num_block = 8 ,sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=2):
        super().__init__()
        self.input_size = input_size # 输入维度
        self.num_block = num_block  # 分块大小
        self.sparsity_threshold = sparsity_threshold  # 那个频率压为0
        self.hard_thresholding_fraction = hard_thresholding_fraction # 低频率取值范围
        self.hidden_size_factor = hidden_size_factor # 进行MLP时hid的映射倍率
        self.block_size = input_size // num_block  # 分块后的特征维度
        self.scale = 0.02

                                # 2 是一个实部一个虚部  中间两个是分块    后面self.hidden_size_factor类似于线性映射
        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_block,self.block_size,self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_block, self.block_size * self.hidden_size_factor))

        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_block,self.block_size*self.hidden_size_factor,self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_block,self.block_size))

    def forward(self, x):
        bias = x

        x = x.float()
        B,H,W,C = x.shape

        x = torch.fft.rfft2(x , dim=(1,2),norm="ortho")    # 二维傅里叶变换 将空间域转化为频率域
        x = x.reshape(B,H,W//2+1,self.num_block,self.block_size)  # W//2+1 是因为对称性压缩频谱 后面可以由前面推出

        # x变为complex形式 想要MLP 就要对实部和虚部分开进行
        o1_real = torch.zeros([B,H,W//2+1,self.num_block,self.block_size * self.hidden_size_factor],device = x.device)
        o1_imag = torch.zeros([B,H,W//2+1,self.num_block,self.block_size * self.hidden_size_factor],device = x.device)
        o2_real = torch.zeros(x.shape,device=x.device)
        o2_imag = torch.zeros(x.shape,device=x.device)


        total_modes = H//2+1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
        # x[...,block,:]@  w1[block,:,:].T
        o1_real[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes].real,self.w1[0])-
            torch.einsum("...bi,bio->...bo", x[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes].imag,self.w1[1])+
            self.b1[0]
        )
        o1_imag[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes].real,self.w1[1])+
            torch.einsum("...bi,bio->...bo", x[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes].imag,self.w1[0])+
            self.b1[1]
        )

        o2_real[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", o1_real[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes],self.w2[0])-
            torch.einsum("...bi,bio->...bo", o1_imag[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes],self.w2[1])+
            self.b2[0]
        )
        o2_imag[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", o1_real[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes],self.w2[1])+
            torch.einsum("...bi,bio->...bo", o1_imag[:,total_modes-kept_modes:total_modes+kept_modes,:kept_modes],self.w2[0])+
            self.b2[1]
        )

        x = torch.stack([o2_real,o2_imag],dim = -1)  # 添加一个新维度 是实部与虚部的区分
        x = F.softshrink(x, lambd=self.sparsity_threshold)  # 小频率压为 0 但是其他非小频率也会进行-self.sparsity_threshold
        x = torch.view_as_complex(x) # 将复数形式的数据合并
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")  # 逆傅里叶变换 转换为空间域
        return x + bias














