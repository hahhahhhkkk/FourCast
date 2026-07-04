class patchembedding(torch.nn.Module):
    def __init__(self, in_channel=24, out_channel=512, patch_size=(2, 2)):
        super().__init__()
        self.c2d = torch.nn.Conv2d(in_channel, out_channel, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(out_channel)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        # 输入B,C,H,W
        data = self.c2d(data)
        data = data.permute(0, 2, 3, 1)
        data = self.norm(data)
        data = self.dropout(data)
        return data.reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[-1])  # 输出B,H,W,C
