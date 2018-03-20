import pandas as pd
import numpy as np

file1 = pd.read_csv("pooled_gru.csv")
file2 = pd.read_csv("bi_lstm_conv.csv")

result = pd.read_csv("data/sample_submission.csv")

cls = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

result[cls] = (np.array(file1[cls]) + np.array(file2[cls])) / 2
result.to_csv("blended_res_pooled_gru_bilstm_conv.csv", index=False)