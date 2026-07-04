from unicodedata import normalize


class Loader(Dataset):
    def __init__(self, data, lead_time,var = None,normalize=True,random_crop=True, noise = True):
        self.lead_time = lead_time
        self.data = data
        self.normalize = normalize
        self.random_crop = random_crop
        self.var = var
        self.noise = noise
        self.position = position
    def __len__(self):
        return len(self.data) - self.lead_time

    def __getitem__(self, idx):
        if self.var is not None:
            self.data = self.data[:,self.var]

        if self.normalize:
            self.data = normalize(self.data)

        if self.random_crop:
            rnd_x = np.random.randint(0,self.data.shape[2]-256)
            rnd_y = np.random.randint(0, self.data.shape[3]-256)
            self.data = self.data[:, :, rnd_x:rnd_x + 256, rnd_y:rnd_y + 256]
        if self.noise:
            self.data = self.data + np.random.normal(0, scale=0.01, size=self.data.shape)

        input_data = torch.tensor(
            self.data[idx],
            dtype=torch.float32
        )
        target_data = torch.tensor(
            self.data[idx + self.lead_time],
            dtype=torch.float32
        )
        return input_data, target_data
