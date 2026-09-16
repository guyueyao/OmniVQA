'''
This is the Code for Training DS-MAE
'''
import logging

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from models import Nomalize, do_maskv3, DSMAE, Decoder0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from myvitimm import myvit_small_patch16_384
import decord

decord.bridge.set_bridge('torch')
from decord import VideoReader, cpu, gpu

p_s = 16


logging.basicConfig(

    level=logging.INFO,  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="SSL.log",  # 日志保存到这个文件
    filemode="a"  # a=追加写入；w=每次运行覆盖旧日志
)
# The frozen encoder, for feature replay
class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.backbone = myvit_small_patch16_384(pretrained=True)
        # self.backbone=torch.jit.script(self.backbone)
        self.a = nn.Parameter(torch.ones((1, 1, 384)) * 0.5)
        self.decoder = Decoder0()

    def forward(self, x):
        with torch.no_grad():
            B, T, _, _, _ = x.shape
            x = x.view(B * T, 3, 384, 384)
            x = self.backbone(x)
        return x

    @torch.no_grad()
    def align(self, x):
        B, T, _, _, _ = x.shape
        x = x.view(B * T, 3, 384, 384)
        feats = self.backbone(x)
        return feats, self.a.detach().clone().squeeze().abs()

    def L1loss(self):
        z = self.a.log() - (1 - self.a).log()
        z = F.sigmoid(z).sum() / 384 * 0.1
        return z


class AUGFrame(Dataset):
    def __init__(self, images):
        self.images = images

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    def __len__(self):
        return len(self.images)

    # def __read_frames(self, vname):
    #     # with open(vname,'rb') as f:
    #     #     vr = VideoReader(f, ctx=cpu())
    #     #     frames=vr[:]
    #     frames = skvideo.io.vread(vname, width=2304, height=384, inputdict={'-pix_fmt': 'yuv420p'})
    #     return frames

    def __read_frames(self,vname):
        with open(vname,'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            frames=vr[:]
        return frames


    '''
    Naming rule:
     folds:  augsm00,augsm01,augsm10,augsm11 contain viewport videos at different locations.
     files: [distortion level:h/m/w]_[distortion type]_[S2P projection type]__[video name].mkv
    '''
    def __getitem__(self, index):
        vname = self.images[index]
        # randomly select one viewport
        i=random.randint(0,1)
        j = random.randint(0, 1)

        #read video frames
        dis = self.__read_frames(
            '/cache/hzy/augsm%d%d/%s' % (i,j,vname))
        vname_ref = vname.split('__')[1]
        ref = self.__read_frames(
            '/cache/hzy/srcsm%d%d/%s' % (i,j,vname_ref))

        c = random.randint(0, 5)
        r = random.randint(0, 6)
        dis = dis[r:r + 4, :, c * 480:c * 480 + 480]
        ref = ref[r:r + 4, :, c * 480:c * 480 + 480]
        dis = dis[:, 48:-48, 48:-48]
        ref = ref[:, 48:-48, 48:-48]
        dis = dis.permute((0, 3, 1, 2)) # T H W C > t c h w
        ref = ref.permute((0, 3, 1, 2))

        # decode distortion type
        cls_info = vname.split('__')[0].split('_')
        d = cls_info[0]
        c = cls_info[1]
        p = cls_info[2]
        d = ['l', 'm', 'h'].index(d)
        c = ['h264', 'hevc'].index(c)
        p = ['e', 'c3x2', 'tsp'].index(p)
        id = d *6+ c*3+p

        sample = {'dis': dis, 'ref': ref, 'label': id}
        return sample  # 返回该样本


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")
    files_train = os.listdir('/cache/hzy/augsm00')
    random.shuffle(files_train)
    train_set = AUGFrame(images=files_train)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=6, pin_memory=True)

    model_t = Teacher().float().to(device)
    model = DSMAE(0.75).float().to(device)

    optimizers = AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5,'weight_decay':0},
        {'params': list(model.decoder.parameters())+list(model.cls.parameters())+[model.mask_token], 'lr': 3e-5,'weight_decay':3e-4}
    ])

    ce_loss = nn.CrossEntropyLoss()
    ssim= nn.L1Loss()
    pixelnorm = Nomalize()

    for epoch in range(0, 101):

        model.train()
        l_t_mse = 0
        l_t_lwf = 0
        l_t_ce = 0
        msk_sum = 0
        bcount=0
        #train only
        for idx, data in enumerate(tqdm(train_loader)):

            label = data['label']
            label = label.view(-1).to(device).long()
            ref = data['ref']
            dis = data['dis']
            ref = ref.to(device).float()
            dis = dis.to(device).float()

            pixelnorm(ref)
            pixelnorm(dis)

            B, T, _, _, _ = dis.shape
            if epoch < 21:
                y, mask1, pree = model.warm_up(ref, dis)
            else:
                y, mask1, pree = model.mae(ref, dis)
            dis_ = do_maskv3(dis.clone(), mask1)
            loss1 = ssim(y, dis_)
            loss5 = F.cross_entropy(pree, label)

            if epoch < 21:
                loss = loss1 + loss5
            else:
                feats = model_t(ref)
                loss3 = model.lwf2(ref, feats)
                loss=loss1+10*loss3+loss5
            loss.backward()

            l_t_mse += loss1.item()
            l_t_ce += loss5.item()
            # ------------------------------------
            bcount += 1
            if bcount >= 2:
                optimizers.step()
                optimizers.zero_grad()
                bcount = 0
        # schedular.step()

        l_t_mse /= idx
        l_t_ce /= idx
        msk_sum /= idx

        tqdm.write(
            'epoch %d > tran loss : MSE %.6f, LWF %.6f, ssim %.6f | mask length %d ' % (epoch, l_t_mse, 0, l_t_ce, msk_sum))

        logging.info(
            'epoch %d > tran loss : MSE %.6f, LWF %.6f, ssim %.6f | mask length %d ' % (epoch, l_t_mse, 0, l_t_ce, msk_sum))

        if epoch % 5 == 0 :
            torch.save({'model0': model.state_dict(),
                        'optim': optimizers.state_dict(),'traintest':files_train
                       },
                       './model/modelPT_' + str(epoch) + '.pth')



