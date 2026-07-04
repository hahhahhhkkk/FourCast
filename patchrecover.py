class patchrevocer(torch.nn.Module):
    def __init__(self,in_channel = 512 ,out_channel = 20,patch_size = (2,2)):
        super().__init__()
        # self.c2d = torch.nn.ConvTranspose2d(in_channel,out_channel,kernel_size=patch_size,stride=patch_size)
        self.pred_head = nn.Sequential(
                # 把 token 通道 D 先压到64，避免内存爆
                nn.Conv2d(in_channel, 64, kernel_size=1, bias=False),
                nn.GELU(),
                nn.GroupNorm(8, 64),

                #
                nn.ConvTranspose2d(64, 64, kernel_size=patch_size, stride=patch_size, bias=False),
                nn.GELU(),
                nn.GroupNorm(8, 64),

                nn.Conv2d(64, out_channel, kernel_size=3, padding=1, bias=True),
            )
        # 网格域细化（3D 卷积，稳定起见用 GroupNorm 而不是 BN）
        self.refine2d = nn.Sequential(
            nn.Conv2d(out_channel, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, out_channel, kernel_size=1),
        )
    def forward(self,data):
        data = data.permute(0,3,1,2)
        data = self.pred_head(data)
        data = self.refine2d(data)
        return data