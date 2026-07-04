up_vars = [
    "U10",
    "V10",
    "T2M",
    "MSL",
    "U1000",
    "U850",
    "U500",
    "U250",
    "V1000",
    "V850",
    "V500",
    "V250",
    "Z1000",
    "Z850",
    "Z500",
    "Z250",
    "T850",
    "Q1000",
    "Q850",
    "Q500"]


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        # x:[T,C,H,W]
        self.mean = x.mean(axis=(0,2,3))
        self.std = x.std(axis=(0,2,3))

    def transform(self, x):
        return (x - self.mean[None,:,None,None]) / self.std[None,:,None,None]

    def inverse_transform(self, x):
        return x * self.std[None,:,None,None] + self.mean[None,:,None,None]