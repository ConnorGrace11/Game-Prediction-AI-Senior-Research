import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sklearn.metrics


statdf = pd.read_csv('qbTotal.csv', header=None)
statdf.columns = ["Date", "Visitors", "V Score", "V Overall", "Home", "H Score", "H Overall", "WinLoss", "V Defense", "H Defense", "V Offense", "H Offense", "V QB", "H QB"]


len(statdf[statdf['WinLoss'] == 1])

team_columns = ["Home", "Visitors"]
stat_columns = ["H Defense", "V Defense", "H Offense", "V Offense", "H Overall", "V Overall", "V QB", "H QB"] #"H Defense", "V Defense", "H Offense", "V Offense", "H Overall", "V Overall", "V QB", "H QB"
x = statdf[stat_columns] #pd.get_dummies(statdf[team_columns+stat_columns], columns=team_columns, prefix=team_columns)
xstats = x.to_numpy()
xmin = xstats.min(axis=0)
xmax = xstats.max(axis=0)
xstats = (xstats - xmin) / (xmax - xmin)
xmeans = xstats.mean(axis=0)
xstats = xstats - xmeans
x[stat_columns] = xstats


print(statdf["WinLoss"].values)
y = pd.Series([val == 1 for val in statdf["WinLoss"].values], index=statdf.index)


indexes = pd.Series(statdf.index).sample(frac=1.0, random_state=0)
train_idxs = list(range(0, int(len(indexes)*0.6)))
val_idxs = list(range(int(len(indexes)*0.6), int(len(indexes)*0.8)))
test_idxs = list(range(int(len(indexes)*0.8), len(indexes)))
trainx = x.iloc[indexes.iloc[train_idxs]] 
valx = x.iloc[indexes.iloc[val_idxs]]
testx = x.iloc[indexes.iloc[test_idxs]]
trainy = y.iloc[indexes.iloc[train_idxs]]
valy = y.iloc[indexes.iloc[val_idxs]]
testy = y.iloc[indexes.iloc[test_idxs]]
torch.tensor(trainy.to_numpy()).long()

testdf = statdf.iloc[indexes.iloc[test_idxs]]
len(testdf[testdf['WinLoss'] == 0])

model = torch.nn.Sequential(
    torch.nn.Linear(trainx.shape[1], 10),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.25),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.25),
    torch.nn.Linear(10, 2),
    torch.nn.Sigmoid()
)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

train_loss_per_epoch = []
train_error_per_epoch = []
val_loss_per_epoch = []
val_error_per_epoch = []
# Training and Test Data
trainx_gpu = torch.tensor(trainx.to_numpy()).float().cuda()
trainy_gpu = torch.tensor(trainy.to_numpy()).long().cuda()
valx_gpu = torch.tensor(valx.to_numpy()).float().cuda()
valy_gpu = torch.tensor(valy.to_numpy()).long().cuda()
testx_gpu = torch.tensor(testx.to_numpy()).float().cuda()
testy_gpu = torch.tensor(testy.to_numpy()).long().cuda()
for epoch in range(500):
    optimizer.zero_grad() 
    train_pred = model(trainx_gpu)
    train_loss = criterion(train_pred, trainy_gpu)
    train_error = 1.0 - sklearn.metrics.accuracy_score([pred[1] > pred[0] for pred in train_pred.cpu().detach().numpy()], trainy)
    val_pred = model(valx_gpu)
    val_loss = criterion(val_pred, valy_gpu)
    val_error = 1.0 - sklearn.metrics.accuracy_score([pred[1] > pred[0] for pred in val_pred.cpu().detach().numpy()], valy)
    print("Train loss: %.4f, val loss: %.4f, train error: %.4f, val error: %.4f" % \
          (train_loss.item(), val_loss.item(), train_error, val_error))
    train_loss_per_epoch.append(train_loss.item())
    train_error_per_epoch.append(train_error)
    val_loss_per_epoch.append(val_loss.item())
    val_error_per_epoch.append(val_error)
    train_loss.backward() 
    optimizer.step() 
test_pred = model(testx_gpu)
test_error = 1.0 - sklearn.metrics.accuracy_score([pred[1] > pred[0] for pred in test_pred.cpu().detach().numpy()], testy)
print("Test error: %.4f" % test_error)

plt.plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, label="train loss")
plt.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, label="val loss")
plt.ylabel('loss',color='b')
plt.xlabel('Number of Epochs',color='b')
plt.legend()
plt.show()

plt.plot(range(len(train_error_per_epoch)), train_error_per_epoch, label="train error")
plt.plot(range(len(val_error_per_epoch)), val_error_per_epoch, label="val error")
plt.ylabel('error',color='b')
plt.xlabel('Number of Epochs',color='b')
plt.legend()
plt.show()

# confusion matrix

cm = sklearn.metrics.confusion_matrix(testy, [pred[1] > pred[0] for pred in test_pred.cpu().detach().numpy()])
cm

cmplot = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=["Win", "Loss"])
cmplot.plot()

cm = sklearn.metrics.confusion_matrix(testy, [pred[1] > pred[0] for pred in test_pred.cpu().detach().numpy()], normalize='true')
cmplot = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=["Loss", "Win"])
cmplot.plot()
