# Data tokenization and pre-processing
from torchtext.data import Dataset


class PTB(Dataset):
    def __init__(self, examples, fields):
        super().__init__(examples, fields)
        pass

    def __len__(self):
        pass


class Yelp(Dataset):
    def __init__(self, examples, fields):
        super().__init__(examples, fields)
        pass

    def __len__(self):
        pass
