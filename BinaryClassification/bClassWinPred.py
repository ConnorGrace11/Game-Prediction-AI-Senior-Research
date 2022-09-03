import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import sklearn.metrics

#load the data and sort it into the relevant columns, as well as normalize the data
statdf = pd.read_csv('qbTotal.csv', header=None)
statdf.columns = ["Date", "Visitors", "V Score", "V Overall", "Home", "H Score", "H Overall", "WinLoss", "V Defense", "H Defense", "V Offense", "H Offense", "V QB", "H QB"]
stat_columns = ["V QB", "H QB"] #"H Defense", "V Defense", "H Offense", "V Offense", "H Overall", "V Overall", "V QB", "H QB"
output = statdf["WinLoss"]
x = statdf[stat_columns]
xstats = x.to_numpy()
xmin = xstats.min(axis=0)
xmax = xstats.max(axis=0)
xstats = (xstats - xmin) / (xmax - xmin)
xmeans = xstats.mean(axis=0)
xstats = xstats - xmeans
x[stat_columns] = xstats

#Load our output values
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

#load test data
testdf = statdf.iloc[indexes.iloc[test_idxs]]
print(len(testdf[testdf['WinLoss'] == 1]))

#initialize the binary classification model
class bClass(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(bClass, self).__init__()
        self.layer_1 = torch.nn.Linear(input_size, 10) 
        self.layer_2 = torch.nn.Linear(10, 10)
        self.layer_out = torch.nn.Linear(10, output_size)         
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.25)
        self.batchnorm1 = torch.nn.BatchNorm1d(10)
        self.batchnorm2 = torch.nn.BatchNorm1d(10)
        
    def forward(self, x):
        output = self.relu(self.layer_1(x))
        output = self.batchnorm1(output)
        output = self.relu(self.layer_2(output))
        output = self.batchnorm2(output)
        output = self.dropout(output)
        output = self.layer_out(output)
        return output
      
      
model = bClass(trainx.shape[1], 2)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#train the binary classification model
train_loss_per_epoch = []
train_error_per_epoch = []
val_loss_per_epoch = []
val_error_per_epoch = []
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

#create loss graph
plt.plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, label="train loss")
plt.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, label="val loss")
plt.ylabel('loss',color='b')
plt.xlabel('Number of Epochs',color='b')
plt.legend()
plt.show()

#create error graph 
plt.plot(range(len(train_error_per_epoch)), train_error_per_epoch, label="train error")
plt.plot(range(len(val_error_per_epoch)), val_error_per_epoch, label="val error")
plt.ylabel('error',color='b')
plt.xlabel('Number of Epochs',color='b')
plt.legend()
plt.show()

#create confusion matrix for test data prediction result
cm = sklearn.metrics.confusion_matrix(testy, [pred[1] > pred[0] for pred in test_pred.cpu().detach().numpy()])

cmplot = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=["Win", "Loss"])
cmplot.plot()
