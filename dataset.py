import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class AMAP_IMG(Dataset):

    def __init__(self, root, train=True, transform = None, target_transform=None):
        super(AMAP_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train :
            file_annotation = root + '/amap_traffic_annotations_train.json'
            img_folder = root + '/amap_traffic_train_0712/'
        else:
            file_annotation = root + '/amap_traffic_annotations_test.json'
            img_folder = root + '/amap_traffic_test_0712/'
        fp = open(file_annotation,'r')
        data_dict = json.load(fp)
        annotations = data_dict['annotations']
        num_data = len(annotations)

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_data):
            self.filenames.append('%06d'%(i+1) + '/' + annotations[i]['key_frame'])
            self.labels.append(annotations[i]['status'])

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]

        img = Image.open(img_name)
        img = img.resize((224, 224), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)

transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = AMAP_IMG('./data',train=True,transform=transform)
test_dataset = AMAP_IMG('./data',train=False,transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)


if __name__ =='__main__':
    for train_batch in train_loader:
        print(train_batch[0].shape)
    for test_batch in test_loader:
        print(test_batch[0].shape)
    # dic = {}
    # for train_batch in train_loader:
    #     _, _, H, w = train_batch[0].shape
    #     dic[H*10000+w] = dic.get(H*10000+w, 0) + 1
    # print(dic) # {7201280: 1199, 7201264: 298, 4800640: 1, 5280960: 2}
