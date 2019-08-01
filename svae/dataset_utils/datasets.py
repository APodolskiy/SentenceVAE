from typing import Optional, Sequence, Tuple

from torchtext.data import Dataset, Field, Example
from tqdm import tqdm


class PTB(Dataset):
    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    name = 'ptb'
    dirname = ''

    def __init__(self,
                 path: str,
                 fields: Sequence[Tuple[str, Field]],
                 max_len: Optional[int] = None,
                 **kwargs):
        examples = []
        with open(path, 'r') as fp:
            for line in tqdm(fp):
                if max_len is not None:
                    line = line[:max_len]
                if line != '':
                    examples.append(Example.fromlist(data=(line, line), fields=fields))
        super().__init__(examples=examples, fields=fields, **kwargs)

    @classmethod
    def splits(cls, fields, max_len=None, root='data', train='ptb.train.txt',
               validation='ptb.valid.txt', test='ptb.test.txt',
               **kwargs):
        return super(PTB, cls).splits(
            root=root, train=train, validation=validation, test=test,
            fields=fields, max_len=max_len
        )


if __name__ == '__main__':
    tokenize = lambda x: x.strip().split()
    text_field = Field(sequential=True, use_vocab=True, init_token='<sos>',
                       eos_token='<eos>', tokenize=tokenize, include_lengths=True)
    print(PTB.splits(fields=(text_field, text_field)))
