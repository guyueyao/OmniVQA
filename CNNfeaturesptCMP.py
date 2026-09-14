
'''
Extract fremlevel features from videos via DS-MAE
'''

import time
import torch
import skvideo.io
import os
import numpy as np
from torch.nn import Module
from tqdm import tqdm
import skvideo.io
import torch.nn as nn
from models import Nomalize, DSMAE
from threading import Thread
import queue

'''
Since video reading is time consuming, videos are read by background threads. This can accelerate the computation.
'''
class VideoReader:
    def __init__(self,videos,video_path,v_length):
        self.videos = videos
        self.v_queue=queue.Queue(len(videos))
        for v in videos:
            self.v_queue.put(v)
        self.root_dir =video_path
        self.vlen= v_length//3
        self.video=queue.Queue(4)
        workers=[]
        for i in range(4):
            worker=Thread(target=self.update_video,daemon=True)
            worker.start()
            workers.append(worker)
    def update_video(self):
        while not self.v_queue.empty():
            vname=self.v_queue.get()
            video = self.video_read(vname)
            if video is not None:
                while self.video.qsize()>=3:
                    time.sleep(0.01)
                self.video.put((video,vname))

    def get_video(self):
        while self.video.empty():
            time.sleep(0.05)
        (video, vname) = self.video.get()
        return video,vname

    def has_video(self):
        return (not self.v_queue.empty()) or (not self.video.empty())
    def video_read(self,vname):
        #
        if (not 'ERP' in vname) and (not 'RCMP' in vname) and (not 'TSP' in vname): # For ODV-VQA dataset only. Skipping reference videos.
            return None
        if not ('.mp4' in vname or '.mov' in vname or '.mkv' in vname or '.mov' in vname):
            return None

        video = np.zeros((self.vlen, 3, 480, 2880), dtype=np.uint8)
        ffmpeg = skvideo.io.FFmpegReader(os.path.join(self.root_dir, vname), #inputdict={'-hwaccel': 'cuda'},
                                         outputdict={'-vf': 'v360=e:c6x1', '-s': '2880x480'})
        fcount = 0
        i=0
        for frame in ffmpeg.nextFrame():
            if not fcount % 3==0:
                fcount+=1
                continue
            frame = np.transpose(frame, [2, 0, 1])  # (H,W,C) -> (C,H,W)
            video[i]=frame
            i+=1
            fcount+=1
            if i>=self.vlen:
                break
        ffmpeg.close()
        return video

# Extract features from all viewports
class vit_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit=DSMAE(0.75)
        self.vit.load_state_dict(torch.load('pretrained/modelPT.pth')['model0'])
        self.norm=Nomalize()
    def forward(self,img):
        # img = img.permute(0, 3, 1, 2)
        img = torch.cat(torch.split(img.unsqueeze(0), 480, dim=-1), dim=0)  # 4 T 3 H W

        img = img[[0,1,4,5], :, :, 48:-48, 48:-48]
        self.norm(img)
        features = self.vit.forward(img)
        return features

if __name__ == "__main__":

    video_path='/home2/ODV-half'
    videos = os.listdir(video_path)
    # parameters
    feature_path = 'features/ODV'
    video_length = 300  # ODV:300  JVQD:120  SVQD:500
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    device = torch.device("cuda")
    extractor = vit_small().bfloat16().to(device)
    extractor.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(videos))
        video_reader = VideoReader(videos, video_path, video_length)
        while video_reader.has_video():
            frames, vname = video_reader.get_video()
            frames = torch.from_numpy(frames).to(device).bfloat16()
            features = extractor(frames)
            features = features.float().data.cpu().numpy()
            np.save(os.path.join(feature_path, vname[:-4] + '.npy'), features)
            pbar.update(1)

