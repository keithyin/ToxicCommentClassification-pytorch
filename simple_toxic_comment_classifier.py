import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer

# from nltk.tokenize import word_tokenize


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

print("train.shape ", train.shape)

cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_pred = pd.read_csv('./data/sample_submission.csv')

for c in cols:
    clf = LogisticRegression(C=4, solver='sag')
    clf.fit(train, y_train[c])
    y_pred[c] = clf.predict_proba(test)[:, 1]
    pred_train = clf.predict_proba(train)[:, 1]
    print(c, '--> log loss:', log_loss(y_train[c], pred_train))

y_pred.to_csv('my_submission.csv', index=False)
