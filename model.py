class model_block(torch.nn.Module):
    def __init__(self, input_size, num_block=8):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(input_size)
        self.afno = AFNO2D(input_size, num_block)
        self.norm2 = torch.nn.LayerNorm(input_size)
        self.linear = MLP(input_size, input_size * 2, input_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        bias = x

        x = self.norm1(x)
        x = self.afno(x)
        x = self.dropout(x)
        x = bias + x
        bias = x
        x = self.norm2(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x + bias


class fourcastmodel(torch.nn.Module):
    def __init__(self, blocks, input_size, num_block=8):
        super().__init__()
        self.block = []
        self.blocks = blocks
        self.pos = position
        self.patchembedding = patchembedding()

        for i in range(blocks):
            self.block.append(model_block(input_size, num_block))
        self.block = nn.ModuleList(self.block)

        self.mlp = MLP(512, 1024, 512)
        self.patctecover = patchrevocer()

    def forward(self, x):
        x = self.pos(x)
        x = self.patchembedding(x)
        for i in range(self.blocks):
            x = self.block[i](x)
        x = self.mlp(x)
        x = self.patctecover(x)
        return x