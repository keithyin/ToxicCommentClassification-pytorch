import spacy
from torchtext import data
from tqdm import tqdm
import pandas as pd
import torch
from torchtext.vocab import Vectors, GloVe
from torch.nn import init

spacy_en = spacy.load('en')


class CustomDataset(data.Dataset):
    name = 'toxic comment'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        csv_data = pd.read_csv(path)
        print("preparing examples...")
        for i in tqdm(range(len(csv_data))):
            if i>100 :
                break
            sample = csv_data.loc[i]
            text, label = self.process_csv_line(sample, test)
            examples.append(data.Example.fromlist([text, label], fields))

        super(CustomDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, root='./data',
               train='train.csv', test='test.csv', **kwargs):
        return super(CustomDataset, cls).splits(
            root=root, train=train, test=test, **kwargs)

    def process_csv_line(self, sample, test):
        text = sample["comment_text"]
        text = text.replace('\n', ' ')
        label = None
        if not test:
            label = [v for v in
                     map(int, sample[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]])]
        return text, label


def prepare_data_and_model(Model, using_gpu=True):
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"

    def tokenize(text):
        fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        trans_map = str.maketrans(fileters, " " * len(fileters))
        text = text.translate(trans_map)
        text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']

        tokenized_text = []
        auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', "'s"]
        for token in text:
            if token == "n't":
                tmp = 'not'
            elif token == "'ll":
                tmp = 'will'
            elif token in auxiliary_verbs:
                tmp = 'be'
            else:
                tmp = token
            tokenized_text.append(tmp)
        return tokenized_text

    TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, fix_length=100, truncate_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True,
                       tensor_type=torch.FloatTensor)
    train = CustomDataset(train_path, text_field=TEXT, label_field=LABEL)
    test = CustomDataset(test_path, text_field=TEXT, label_field=None, test=True)
    vectors = GloVe(name='6B', dim=300)
    vectors.unk_init = init.xavier_uniform
    TEXT.build_vocab(train, vectors=vectors, max_size=30000)

    print('train.fields', train.fields)
    print('train.name', getattr(train, 'text'))
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))

    # using the training corpus to create the vocabulary

    train_iter = data.Iterator(dataset=train, batch_size=32, train=True, repeat=False,
                               device=0 if using_gpu else -1)
    test_iter = data.Iterator(dataset=test, batch_size=64, train=False, sort=False, device=0 if using_gpu else -1)

    num_tokens = len(TEXT.vocab.itos)
    num_classes = 6

    net = Model(embedding_size=300, num_tokens=num_tokens, num_classes=num_classes)
    net.embedding.weight.data.copy_(TEXT.vocab.vectors)
    if using_gpu:
        net.cuda()

    return train_iter, test_iter, net
