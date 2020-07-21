# %%
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from dataset import train_loader, test_loader


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 3) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# %%
epo_num = 10
savePath = './checkpoints/'
if not os.path.exists(savePath):
    os.makedirs(savePath)
modelName = 'resnet50'

for epo in range(epo_num):
    model.train()
    for index, (imgs, labels) in enumerate(train_loader):
        preds = model(imgs.to(device))
        loss = criterion(preds, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            iter_loss = loss.item()
            print('epoch {}, {:03d}/{},train loss is {:.4f}'.format(epo, index, len(train_loader), iter_loss), end="\n")

    if epo % 1 == 0:
        torch.save(model, savePath + modelName + '_{}.pt'.format(epo))
        print('saveing ' + savePath + modelName + '_{}.pt'.format(epo))

# %%
