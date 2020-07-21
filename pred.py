# %%
import os
import torch
import torch.nn as nn
import torchvision.models as models
from dataset import train_loader, test_loader


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./checkpoints/resnet50_7.pt')
model = model.to(device)
model.eval()

# %%
predict = []
for index, (imgs, _) in enumerate(test_loader):
    if index % 100 == 0:
        print(index)
    with torch.no_grad():
        preds = model(imgs.to(device))
        idx = torch.argmax(preds, 1)
        predict += list(idx.cpu().numpy())

# %%
import json

json_path = "data/amap_traffic_annotations_test.json"
out_path = "data/amap_traffic_annotations_test_result.json"

# result 是你的结果, key是id, value是status
with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
    json_dict = json.load(f)
    data_arr = json_dict["annotations"]  
    new_data_arr = [] 
    for data, pred in zip(data_arr, predict):
        data["status"] =int(pred)
        new_data_arr.append(data)
    json_dict["annotations"] = new_data_arr
    json.dump(json_dict, w)

# %%
