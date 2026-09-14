'''
code for intra-dataset evaluation
'''

import sys
import os
import torch
from torch.optim import Adam, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
from datasets.DataSource import ODVSource,JVQDSource,SVQDSource

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
            feature=np.load(os.path.join(self.fpath,vname[:-4]+'.npy'))
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
    # parameters
    Videosource=ODVSource
    feature_path='features/ODV'
    video_length=300 #ODV: 300  JVQD:120 SVQD: 500
    # torch.set_float32_matmul_precision('high')
    #
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    videosource = Videosource() # generate tran-test splits
    bsrcc=np.zeros((5,1))
    bplcc = np.zeros((5, 1))
    brmse = np.zeros((5, 1))

    for r in range(0,5):
        rounds_index = r
        checkpoint = videosource.fiveFolds[r] # 5-folds eval
        train_images = checkpoint['train_images']
        train_dmos = checkpoint['train_dmos']
        test_images = checkpoint['test_images']
        test_dmos = checkpoint['test_dmos']

        train_set = VideoFeat( images=train_images, dmos=train_dmos,fpath=feature_path,vlengh=video_length)
        test_set = VideoFeat( images=test_images, dmos=test_dmos,fpath=feature_path,vlengh=video_length)
        dataloader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4,pin_memory=True)
        testloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4,pin_memory=True)

        model = OVQA(384).float().to(device)
        criterion = nn.MSELoss()

        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sroccbest=0
        for epoch in range(500):
            # ------train-------------
            model.train()
            L = 0
            for idx, data in enumerate((dataloader)):
                features=data['feat']
                label=data['label']
                features = features.to(device).float()
                features=features.squeeze(1)
                label = label.to(device).float()
                optimizer.zero_grad()  #
                outputs = model(features)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                L = L + loss.item()
            train_loss = L / (idx + 1)

            # ------eval-------------
            model.eval()
            pre = np.array([0])
            tar = np.array([0])
            mydic=dict()
            with torch.no_grad():
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

            if srocc1>sroccbest:
                sroccbest=srocc1
                bsrcc[r]=srocc1
                bplcc[r]=plcc1
                brmse[r]=rmse1
                checkpoint={"model_state_dict": model.state_dict()}
                # torch.save(checkpoint,'./model/model_'+str(rounds_index)+'.pth')
                print('epoch %d, plcc %.4f , srocc %.4f ,rmse %.4f,%.4f' % (r,plcc1, srocc1, rmse1,sroccbest))

    print(np.mean(bsrcc))
    print(np.mean(bplcc))
    print(np.mean(brmse))