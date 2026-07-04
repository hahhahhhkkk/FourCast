class MLP(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, act_factor=nn.GELU(), drop_rate=0):
        super().__init__()
        self.liner1 = nn.Linear(in_dim, hid_dim)
        self.liner2 = nn.Linear(hid_dim, out_dim)
        self.act = act_factor  # 激活层为网络引入非线性
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.drop(self.act(self.liner1(x)))
        x = self.drop((self.liner2(x)))
        return x
