def reshape_fields(img, inp_or_tar, crop_size_x, crop_size_y, rnd_x, rnd_y, y_roll=True, train=True, add_grid=True,
                   gridtype="sinusoidal", normalize=True, two_step_training=False, add_noise=False):
    # 输入（T,C,H,W）
    # 返回（T*C，crop_size_x,crop_size_y）
    img = img[:, :, 0:48]
    T, C, H, W = img.shape
    means = np.load("mean.npy")[:, :C]
    std = np.load("std.npy")[:, :C]

    if normalize:  # 添加归一化
        img = img - means
        img = img * std

    if add_grid:  # 添加位置编码
        if inp_or_tar == "inp":
            if gridtype == "linear":
                grid_x = np.meshgrid(np.linspace(-1, 1, H))
                grid_y = np.meshgrid(np.linspace(-1, 1, w))
                grid_x, grid_y = np.meshgrid(grid_y, grid_x)
                grid = np.stack([grid_x, grid_y], axis=0)
                C = C + 2
            else:
                x1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, H)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, H)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, W)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, W)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack([grid_x1, grid_y1, grid_x2, grid_y2], axis=0), axis=0)
                C = C + 4
        img = np.concatenate([img, grid.repeat(32, axis=0)], axis=1)

    # 添加地形
    orog = np.load("dixing.npy")
    img = np.concatenate([img, np.expand_dims(orog, axis=(0, 1)).repeat(32, axis=0)], axis=1)
    C = C + 1

    if y_roll:  # 对W进行轻微位移
        img = np.roll(img, 2, axis=-1)

    # 根据起点和给出的图像大小 进行剪裁
    if train and (crop_size_x or crop_size_y):
        img = img[:, :, rnd_x:rnd_x + crop_size_x, rnd_y:rnd_y + crop_size_y]
    if inp_or_tar == "inp":  # 对输入数据操作
        img = img.reshape(T * C, crop_size_x, crop_size_y)
    elif inp_or_tar == "tar":  # 对目标值操作
        if two_step_training:
            img = img.reshape(C * 2, crop_size_x, crop_size_y)
        else:
            img = img.reshape(C, crop_size_x, crop_size_y)

    if add_noise:
        img = img + np.random.normal(0, scale=0.01, size=img.shape)

    return torch.Tensor(img)



