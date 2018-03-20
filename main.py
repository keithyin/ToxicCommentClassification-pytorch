from torch import nn
from torch import optim
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import data_preprocess
from nets import models


def accuracy(logits, labels):
    """
    calculate the accuracy
    :param logits: Variable [batch_size, ]
    :param labels: Variable [batch_size]
    :return:
    """
    logits = logits.data
    labels = labels.data
    logits[logits > .5] = 1
    logits[logits <= .5] = 0

    return torch.mean((logits.long() == labels.long()).type(torch.cuda.FloatTensor))


if __name__ == '__main__':
    submission = pd.read_csv('./data/sample_submission.csv')

    Model = models.BiLSTMConv

    train_iter, test_iter, net = data_preprocess.prepare_data_and_model(Model=Model, using_gpu=True)

    net.optimizer = optim.Adam(params=net.parameters(), lr=1e-3, weight_decay=1e-4)
    net.lr_scheduler = optim.lr_scheduler.StepLR(net.optimizer, step_size=1000, gamma=.99)
    net.loss_func = nn.BCELoss()

    for epoch in range(10):
        train_accs = []
        for batch in tqdm(train_iter):
            net.lr_scheduler.step()

            net.train()
            xs = batch.text
            ys = batch.label

            logits = net(xs)
            logits = torch.sigmoid(logits)

            loss = net.loss_func(logits, ys)

            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()

            train_accs.append(accuracy(logits, labels=ys))

        print("epoch {} :  training accumulated accuracy {} ".format(epoch, np.mean(train_accs)))
        torch.save(net.state_dict(), "{}.pkl".format(Model.name))

        test_accs = []
        net.eval()

        results = []
        print("running testing.....")
        for batch in tqdm(test_iter):
            xs = batch.text
            logits = net(xs)
            prob = torch.sigmoid(logits)
            results.append(prob)

        results = torch.cat(results, dim=0)
        results = results.data.cpu().numpy()

        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = results

        submission.to_csv("{}.csv".format(Model.name), index=False)
        print("*****************result saved****************************************")
