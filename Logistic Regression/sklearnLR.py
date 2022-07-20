from sklearn.linear_model import LogisticRegression
import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sklearn.metrics

statdf = pd.read_csv('total.csv', header=None)
statdf.columns = ["Date", "Visitors", "V Score", "V Overall", "Home", "H Score", "H Overall", "WinLoss", "V Defense", "H Defense", "V Offense", "H Offense"]

print(len(statdf[statdf['WinLoss'] == 1]))

team_columns = ["Home"]
stat_columns = ["H Overall", "V Overall", "H Offense", "V Offense", "H Defense", "V Defense"]
x = pd.get_dummies(statdf[team_columns+stat_columns], columns=team_columns, prefix=team_columns)
xstats = x[stat_columns].to_numpy()
xmin = xstats.min(axis=0)
xmax = xstats.max(axis=0)
xstats = (xstats - xmin) / (xmax - xmin)
xmeans = xstats.mean(axis=0)
xstats = xstats - xmeans
x[stat_columns] = xstats

print(statdf["WinLoss"].values) # notice the extra space in the values
y = pd.Series([val == 1 for val in statdf["WinLoss"].values], index=statdf.index) # keep adultdf index for train/test splits

indexes = pd.Series(statdf.index).sample(frac=1.0, random_state=0)
train_idxs = list(range(0, int(len(indexes)*0.6)))
val_idxs = list(range(int(len(indexes)*0.6), int(len(indexes)*0.8)))
test_idxs = list(range(int(len(indexes)*0.8), len(indexes)))
trainx = x.iloc[indexes.iloc[train_idxs]] # be careful to use iloc twice, otherwise will just grab indexes 0, 1, ... for train
valx = x.iloc[indexes.iloc[val_idxs]]
testx = x.iloc[indexes.iloc[test_idxs]]
trainy = y.iloc[indexes.iloc[train_idxs]]
valy = y.iloc[indexes.iloc[val_idxs]]
testy = y.iloc[indexes.iloc[test_idxs]]
torch.tensor(trainy.to_numpy()).long()

testdf = statdf.iloc[indexes.iloc[test_idxs]]
print(len(testdf[testdf['WinLoss'] == 1]))

trainx_gpu = torch.tensor(trainx.to_numpy()).float()
trainy_gpu = torch.tensor(trainy.to_numpy()).long()
testx_gpu = torch.tensor(testx.to_numpy()).float()
testy_gpu = torch.tensor(testy.to_numpy()).long()
clf = LogisticRegression().fit(trainx_gpu, trainy_gpu)
print(clf.predict_proba(testx_gpu))

print(clf.predict(testx_gpu))
print(testy_gpu)

print(clf.score(testx_gpu, testy_gpu))
