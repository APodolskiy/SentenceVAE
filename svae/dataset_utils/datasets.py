import os
import random
import re
from typing import Optional, Sequence, Tuple, Union, List

from tqdm import tqdm

from torchtext.utils import unicode_csv_reader
from torchtext.data import Dataset, Field, Example


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
                    line_parts = line.split()
                    truncated_line_parts = line_parts[:max_len]
                    line = ' '.join(truncated_line_parts)
                if line != '':
                    examples.append(Example.fromlist(data=(line, line), fields=fields))
        super().__init__(examples=examples, fields=fields, **kwargs)

    @classmethod
    def splits(cls, fields, max_len=None, root='data', train='ptb.train.txt',
               validation='ptb.valid.txt', test='ptb.test.txt',
               **kwargs):
        return super(PTB, cls).splits(
            root=root, train=train, validation=validation, test=test,
            fields=fields, max_len=max_len, **kwargs
        )


class YelpReview(Dataset):
    urls = ['https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0']
    name = 'yelp_review'
    dirname = ''

    def __init__(self,
                 path: str,
                 fields: Sequence[Tuple[str, Field]],
                 num_samples: Optional[int] = None,
                 add_cls: bool = False,
                 random_state: int = 162,
                 max_len: Optional[int] = None,
                 **kwargs):
        duplicate_spaces_re = re.compile(r' +')
        with open(path, 'r', encoding='utf-8') as fp:
            all_data = []
            reader = unicode_csv_reader(fp)
            for row in reader:
                cls, text = row[0], row[1]
                if max_len is not None and len(text.split()) > max_len:
                    continue
                text = text.replace('\\n\\n', '\\n')
                text = duplicate_spaces_re.sub(' ', text)
                data = (text, text, cls) if add_cls else (text, text)
                all_data.append(data)
        if num_samples is not None and num_samples < len(all_data):
            random.seed(random_state)
            all_data = random.sample(all_data, num_samples)
        examples = []
        for data in tqdm(all_data, desc='Converting data into examples'):
            examples.append(Example.fromlist(data=data, fields=fields))
        super().__init__(examples=examples, fields=fields, **kwargs)

    @classmethod
    def splits(cls,
               fields: Sequence[Tuple[str, Field]],
               root: str = 'data',
               split_ratio: Union[float, List[float]] = 0.7,
               stratified: bool = False,
               strata_field: str = 'label',
               num_samples: Optional[int] = None,
               add_cls: bool = False,
               random_state: int = 162,
               max_len: Optional[int] = None,
               **kwargs):
        path = os.path.join(root, cls.name, 'train.csv')
        full_dataset = YelpReview(path=path, fields=fields, num_samples=num_samples,
                                  add_cls=add_cls, random_state=random_state, max_len=max_len, **kwargs)
        splitted_data = full_dataset.split(split_ratio=split_ratio, stratified=stratified, strata_field=strata_field)
        return splitted_data


if __name__ == '__main__':
    tokenize = lambda x: x.strip().split()
    text_field = Field(sequential=True, use_vocab=True, init_token='<s>',
                       eos_token='</s>', tokenize=tokenize, include_lengths=True)
    print(PTB.splits(fields=(('inp', text_field), ('trg', text_field))))
