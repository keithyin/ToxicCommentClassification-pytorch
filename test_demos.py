import pandas as pd

# test_path = "./data/test.csv"
#
# datas = pd.read_csv(test_path)
#
# datas = datas["comment_text"]
#
# # for i, data in enumerate(datas):
# #     data = data.replace('\n', '')
# #     print(i, " --- ", data)

import spacy

spacy_en = spacy.load('en')

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


if __name__ == '__main__':
    a = "you son of a bitch. man don't do that, i'll, that's good"

    print(tokenize(a))
