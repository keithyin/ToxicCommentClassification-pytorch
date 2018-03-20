import pandas as pd
import re
from torch.autograd import Variable
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.nn import init
from torch import optim
import numpy as np


class MLP(nn.Module):
    def __init__(self, feat_size):
        super(MLP, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(in_features=feat_size, out_features=20),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.2),
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=10, out_features=6)
        )

    def forward(self, x):
        logits = self.pipeline(x)
        probs = torch.sigmoid(logits)
        return probs


def prepocess():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    print(train.shape, test.shape)

    train_rows = train.shape[0]

    y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    train.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)

    data = pd.concat([train, test])
    del train
    del test

    # stop_words = set(nltk.corpus.stopwords.words('english'))

    def preprocess_input(comment):
        comment = comment.strip()
        comment = comment.lower()
        words = re.split('\s', comment)
        # words = [word for word in words if not word in stop_words]
        sentence = ' '.join(words)
        return sentence

    data.comment_text = data.comment_text.apply(lambda row: preprocess_input(row))
    # from nltk.tokenize import word_tokenize

    """min_df: 3--->10; result:.9724--->.9735"""
    vect = TfidfVectorizer(min_df=10, max_df=0.7,
                           analyzer='word',
                           ngram_range=(1, 2),
                           strip_accents='unicode',
                           smooth_idf=True,
                           sublinear_tf=True,
                           max_features=30000
                           )

    vect = vect.fit(data['comment_text'])

    # print(vect.vocabulary_)
    # print(len(vect.vocabulary_))
    # exit()

    data_tranformed = vect.transform(data['comment_text'])
    test = data_tranformed[train_rows:]
    train = data_tranformed[:train_rows]
    y_train = np.array(y_train).astype(np.float32)

    return train, y_train, test

    # print("train.shape ", train.shape)
    #
    #
    # cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # y_pred = pd.read_csv('./data/sample_submission.csv')
    #
    # for c in cols:
    #     clf = LogisticRegression(C=4, solver='sag')
    #     clf.fit(train, y_train[c])
    #     y_pred[c] = clf.predict_proba(test)[:,1]
    #     pred_train = clf.predict_proba(train)[:,1]
    #     print(c, '--> log loss:', log_loss(y_train[c], pred_train))
    #
    #
    # y_pred.to_csv('my_submission.csv', index=False)


def init_param(self):
    if isinstance(self, (nn.Conv2d, nn.Linear)):
        init.xavier_uniform(self.weight.data)
        init.constant(self.bias.data, 0)


def batch_generator(batch_size, batch_x, batch_y=None, shuffle=True):
    num_examples = batch_x.shape[0]
    indices = list(range(num_examples))
    if shuffle:
        np.random.shuffle(indices)
    counter = 0
    mini_batch_x = []
    mini_batch_y = []
    for idx in indices:
        mini_batch_x.append(batch_x[idx].toarray())
        if batch_y is not None:
            mini_batch_y.append(batch_y[idx])
        counter += 1
        if counter == batch_size:
            counter = 0
            xs = np.concatenate(mini_batch_x, axis=0)
            ys = None if batch_y is None else np.stack(mini_batch_y, axis=0)
            yield xs, ys
            mini_batch_x = []
            mini_batch_y = []

    if len(mini_batch_x) > 0:
        xs = np.concatenate(mini_batch_x, axis=0)
        ys = None if batch_y is None else np.stack(mini_batch_y, axis=0)
        yield xs, ys


if __name__ == '__main__':
    # prepare data
    train, y_train, test = prepocess()

    mlp = MLP(feat_size=30000)
    mlp.apply(init_param)
    mlp.cuda()
    optimizer = optim.SGD(params=mlp.parameters(), lr=3e-3, momentum=.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=.9)

    criterion = nn.BCELoss()
    min_loss = 10
    for i in range(100):
        losses = []
        mlp.train()
        print("training....")
        for xs, ys in batch_generator(100, train, y_train):
            xs = Variable(torch.FloatTensor(xs).cuda())
            ys = Variable(torch.FloatTensor(ys).cuda())
            prob = mlp(xs)
            loss = criterion(prob, ys)
            losses.append(loss.data)
            lr_scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_ = torch.mean(torch.cat(losses))
        print("epoch {}, loss {}".format(i, loss_))
        mlp.eval()

        probs = []
        print("testing...")
        for xs, _ in batch_generator(100, test):
            xs = Variable(torch.FloatTensor(xs).cuda(), volatile=True)
            prob = mlp(xs)
            probs.append(prob.data)
        probs = torch.cat(probs, dim=0).cpu().numpy()
        print(probs.shape)
        predictions = pd.read_csv("./data/sample_submission.csv")

        for j, c in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
            predictions[c] = probs[:, j]
        if loss_ < min_loss:
            min_loss = loss_
            predictions.to_csv("submission_tf_idf_mlp_%d.csv" % i, index=False)
        print("submission saved!")
        print("******************************************")
