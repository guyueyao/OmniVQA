import os
import torch
import skvideo.io
import sys
import numpy as np
import skvideo.io
import torch.nn as nn

from CrossSetEval import Mlp
from models import Nomalize, DSMAE

def video_read( vname):

    video=skvideo.io.vread(vname,  # inputdict={'-hwaccel': 'cuda'},
                                     outputdict={'-vf': 'v360=e:c6x1', '-s': '2880x480'})
    video=video[0::3]
    video = np.transpose(video, [0, 3, 1, 2])
    return video

class vit_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit=DSMAE(0.75)
        self.vit.load_state_dict(torch.load('pretrained/modelPT.pth')['model0'])
        self.norm=Nomalize()
    def forward(self,img):
        # img = img.permute(0, 3, 1, 2)
        img = torch.cat(torch.split(img.unsqueeze(0), 480, dim=-1), dim=0)  # 4 T 3 H W

        img = img[:, :, :, 48:-48, 48:-48]
        self.norm(img)
        features = self.vit.forward(img)
        return features

class OVQA(nn.Module):
    def __init__(self,dim_in=384):
        super().__init__()
        self.head=Mlp(384,384,1)
    def forward(self,x):
        q = self.head(x).mean(dim=2).mean(dim=1)
        return q.view(-1)

def inference(video_path):
    device = torch.device("cuda")
    frames=video_read(video_path)
    frames = torch.from_numpy(frames).to(device).bfloat16()

    extractor=vit_small().bfloat16().to(device)
    features = extractor(frames).unsqueeze(0).float()

    model = OVQA().float().to(device)
    model.load_state_dict(torch.load('./weights/modelPT.pth'))
    model.eval()
    output = model(features)
    output = output.to('cpu').item()
    return 1-output

if __name__ == "__main__":

    video_file = sys.argv[1]
    score=inference(video_file)
    print('quality prediction: %.6f'%score)