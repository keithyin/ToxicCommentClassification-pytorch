from torch import nn
import torch
from torch.nn import init

__all__ = ['BiGRU', 'WordCNN', 'BiGRUWithTimeDropout', 'BiLSTMConv']


class WordCNN(nn.Module):
    name = "WordCNN"

    def __init__(self, embedding_size, num_tokens, num_classes):
        super(WordCNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_size)

        self.branch_a_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, embedding_size), padding=(1, 0)),
            nn.ReLU(inplace=True)
        )
        self.branch_a_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, embedding_size), padding=(2, 0)),
            nn.ReLU(inplace=True)
        )

        self.branch_a_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(7, embedding_size), padding=(3, 0)),
            nn.ReLU(inplace=True)
        )

        self.post_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flattern(),
            nn.Dropout(),
            nn.Linear(in_features=300, out_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=100, out_features=num_classes)
        )
        self.apply(WordCNN.init_param)

    def forward(self, x):
        """x: lookup indices"""
        x = self.embedding(x)

        x = torch.unsqueeze(x, dim=1)

        x1 = self.branch_a_1(x)
        x2 = self.branch_a_2(x)
        x3 = self.branch_a_3(x)

        x = torch.cat([x1, x2, x3], dim=1)

        return self.post_block(x)

    @staticmethod
    def init_param(self):
        if isinstance(self, (nn.Conv2d, nn.Linear)):
            init.xavier_uniform(self.weight.data)
            init.constant(self.bias.data, 0)


class BiGRU(nn.Module):
    name = "BiGRU"

    def __init__(self, embedding_size, num_tokens, num_classes):
        super(BiGRU, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_size)
        # [batch, len, channel]
        self.spatial_dropout = nn.Dropout2d(p=0.2)
        self.bi_gru = nn.GRU(input_size=300, hidden_size=80, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=320, out_features=num_classes)

        self.apply(BiGRU.init_param)

    def forward(self, x):
        """x: lookup indices"""
        # [batch_size, len, channel]
        x = self.embedding(x)
        x = torch.unsqueeze(x.permute(0, 2, 1), dim=-1)
        x = self.spatial_dropout(x)
        # [b, len, channel]
        x = torch.squeeze(x, dim=-1).permute(0, 2, 1)
        output, _ = self.bi_gru(x)
        avg_pooling = torch.mean(output, dim=1)
        max_pooling, _ = torch.max(output, dim=1)
        feat = torch.cat((avg_pooling, max_pooling), dim=1)
        logits = self.fc(feat)
        return logits

    @staticmethod
    def init_param(self):
        if isinstance(self, (nn.Conv2d, nn.Linear)):
            init.xavier_uniform(self.weight.data)
            init.constant(self.bias.data, 0)


class BiGRUWithTimeDropout(nn.Module):
    name = "BiGRUWithTimeDropout"

    def __init__(self, embedding_size, num_tokens, num_classes):
        super(BiGRUWithTimeDropout, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_size)
        # [batch, len, channel]
        self.time_dropout = nn.Dropout2d(p=0.2)
        self.bi_gru = nn.GRU(input_size=300, hidden_size=80, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=320, out_features=num_classes)

        self.apply(BiGRU.init_param)

    def forward(self, x):
        """x: lookup indices"""
        # [batch_size, len, channel]
        x = self.embedding(x)
        x = torch.unsqueeze(x, dim=-1)
        x = self.time_dropout(x)
        # [b, len, channel]
        x = torch.squeeze(x, dim=-1)
        output, _ = self.bi_gru(x)
        avg_pooling = torch.mean(output, dim=1)
        max_pooling, _ = torch.max(output, dim=1)
        feat = torch.cat((avg_pooling, max_pooling), dim=1)
        logits = self.fc(feat)
        return logits

    @staticmethod
    def init_param(self):
        if isinstance(self, (nn.Conv2d, nn.Linear)):
            init.xavier_uniform(self.weight.data)
            init.constant(self.bias.data, 0)


class BiLSTMConv(nn.Module):
    name = "BiLSTMConv"

    def __init__(self, embedding_size, num_tokens, num_classes):
        super(BiLSTMConv, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_size)
        # [batch, len, channel]
        self.spatial_dropout = nn.Dropout2d(p=0.2)
        self.bi_lstm = nn.LSTM(input_size=300, hidden_size=128, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(in_channels=128 * 2, out_channels=64, kernel_size=3)
        self.fc = nn.Linear(in_features=64 * 2, out_features=num_classes)

        self.apply(BiGRU.init_param)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.unsqueeze(x.permute(0, 2, 1), dim=-1)
        x = self.spatial_dropout(x)
        # [b, len, channel]
        x = torch.squeeze(x, dim=-1).permute(0, 2, 1)

        # output, [bs, len, hidden_size*2]
        x, _ = self.bi_lstm(x)

        # x [bs, 64, len-2]
        x = self.conv1(x.permute(0, 2, 1))
        # average through timestep axis
        avg_pooling = torch.mean(x, dim=2)
        max_pooling, _ = torch.max(x, dim=2)
        feat = torch.cat((avg_pooling, max_pooling), dim=1)

        logits = self.fc(feat)
        return logits

    @staticmethod
    def init_param(self):
        if isinstance(self, (nn.Conv2d, nn.Linear)):
            init.xavier_uniform(self.weight.data)
            init.constant(self.bias.data, 0)


class Flattern(nn.Module):
    def __init__(self):
        super(Flattern, self).__init__()

    def forward(self, x):
        bs = x.size(0)
        return x.view(bs, -1)
