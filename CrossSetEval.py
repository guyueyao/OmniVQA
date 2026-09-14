'''
code for interdataset evaluation
'''

import os.path
import torch
from timm.layers import Mlp
from torch.optim import Adam, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
from datasets.DataSource import ODVSource, JVQDSource, SVQDSource, VRVQWSource
import torch.nn.functional as F

class VideoFeat(Dataset):
    def __init__(self, images, dmos,fpath, vlengh):
        self.images = images
        self.dmos = dmos
        self.vlen=vlengh//3-2
        self.fpath=fpath
        self.preload()


    def __len__(self):
        return len(self.images)

    def preload(self):
        self.features=[]
        self.dmoses=[]
        for i in range(len(self.images)):
            vname = self.images[i]
            feature=np.load(os.path.join(self.fpath,vname[:-4]+'.npy'))[:,:,:self.vlen,:]
            dmos=self.dmos[i]
            self.features.append(feature)
            self.dmoses.append(dmos)

    def __getitem__(self, index):
        vname = self.images[index]
        features=self.features[index]
        dmos=self.dmoses[index]
        sample = {'feat': features, 'label': dmos,'name':vname}
        return sample

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(0.2)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

class OVQA(nn.Module):
    def __init__(self,dim_in=384):
        super().__init__()
        self.head=Mlp(384,384,1)
    def forward(self,x):
        q = self.head(x).mean(dim=2).mean(dim=1)
        return q.view(-1)

if __name__ == "__main__":
    #parameters
    SrcSource=VRVQWSource
    TarSource=JVQDSource
    feature_path_src = './features/VRVQW'
    video_length_src = 300  # ODV: 300  JVQD:120 SVQD: 500
    feature_path_tar = './features/JVQD'
    video_length_tar = 120  # ODV: 300  JVQD:120 SVQD: 500

    #
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #
    device = torch.device("cuda")
    # get the train samples from source dataset
    srcsource=SrcSource()
    checkpoint = srcsource.fiveFolds[0]
    train_images = checkpoint['train_images']
    train_dmos = checkpoint['train_dmos']
    test_images = checkpoint['test_images']
    test_dmos = checkpoint['test_dmos']
    train_images=train_images+test_images
    train_dmos=np.concatenate((train_dmos,test_dmos))
    
    bsrcc=np.zeros((5,1))
    bplcc = np.zeros((5, 1))
    brmse = np.zeros((5, 1))

    # get the test samples from target dataset
    videosource=TarSource()
    checkpoint = videosource.fiveFolds[0]
    train_images_ = checkpoint['train_images']
    train_dmos_ = checkpoint['train_dmos']
    test_images = checkpoint['test_images']
    test_dmos = checkpoint['test_dmos']
    test_images = train_images_ + test_images
    test_dmos = np.concatenate((train_dmos_, test_dmos))
    #
    train_set = VideoFeat( images=train_images, dmos=train_dmos,fpath=feature_path_src,vlengh=video_length_src)
    test_set = VideoFeat( images=test_images, dmos=test_dmos,fpath=feature_path_tar,vlengh=video_length_tar)
    dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2,pin_memory=True)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2,pin_memory=True)

    model = OVQA(384).float().to(device)
    criterion = nn.MSELoss() 
    optimizer = AdamW(model.parameters(), lr=3e-5)
    sroccbest=0

    for epoch in range(500):
        model.train()
        L = 0
        for idx, data in enumerate((dataloader)):
            features=data['feat']
            label=data['label']
            features = features.to(device).float()
            features = features.squeeze(1)
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (idx + 1)


        model.eval()
        pre = np.array([0])
        tar = np.array([0])
        mydic=dict()
        for idx, data in enumerate((testloader)):
            features=data['feat']
            label=data['label']
            features = features.to(device).float()
            features = features.squeeze(1)
            label = label.data.numpy().flatten()
            output = model(features)
            output = output.to('cpu')
            predict = output.data.numpy().flatten()
            pre = np.hstack((pre, predict))
            tar = np.hstack((tar, label))

        srocc1, _ = stats.spearmanr(pre[1:], tar[1:]) 
        plcc1, _ = stats.pearsonr(pre[1:], tar[1:])
        rmse1 = np.sqrt(np.mean(np.square(pre[1:] - tar[1:])))

        if abs(srocc1)>sroccbest:
            sroccbest=abs(srocc1)
            print('epoch %d,  srocc %.4f, plcc %.4f, rmse %.4f, %.4f' % (epoch, abs(srocc1),abs(plcc1),rmse1,sroccbest))
