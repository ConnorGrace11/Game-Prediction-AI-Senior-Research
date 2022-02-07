import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Using data from the Pittsburgh Steelers 2021 season
games = list(range(1, 18))
games_train = torch.tensor(np.array(games, dtype=np.float32))
games_train = games_train.reshape(-1, 1)

scores = [23, 17, 10, 17, 27, 23, 15, 29, 16, 37, 10, 20, 28, 19, 10, 26, 16]
scores_train = torch.tensor(np.array(scores, dtype=np.float32))
scores_train = scores_train.reshape(-1, 1)

games_train

scores_train

class linearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        
    def forward(self, x):
        output = self.linear(x)
        return output


model = linearRegression (1, 1)
#model.cuda()#put on GPU


for p in model.parameters():
    print(p.data)


criterion = torch.nn.MSELoss() 

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


for epoch in range(100):
    pred = model(games_train)
    optimizer.zero_grad() 
    loss = criterion(pred, scores_train)
    print(loss.item())
    loss.backward() 
    optimizer.step() 

plt.scatter(games_train.cpu(), scores_train.cpu(), label='true')
plt.scatter(games_train, pred.detach().numpy(), label = 'pred')
plt.show()
