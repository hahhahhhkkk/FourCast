# 位置编码 直接添加到特征维度
def position(img):
    B,C,H,W = img.shape
    x1 = np.meshgrid(np.sin(np.linspace(0,2*np.pi,H)))
    x2 = np.meshgrid(np.cos(np.linspace(0,2*np.pi,H)))
    y1 = np.meshgrid(np.sin(np.linspace(0,2*np.pi,W)))
    y2 = np.meshgrid(np.cos(np.linspace(0,2*np.pi,W)))
    grid_x1 ,grid_y1 = np.meshgrid(y1,x1)
    grid_x2 ,grid_y2 = np.meshgrid(y2,x2)
    grid = np.expand_dims(np.stack([grid_x1,grid_y1,grid_x2,grid_y2],axis = 0),axis = 0)
    img = torch.concatenate([img,torch.Tensor(grid.repeat(B,axis = 0)).to(device)],axis = 1)
    return img