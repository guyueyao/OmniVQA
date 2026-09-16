'''
Model defintion of DS-MAE
'''

import random
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models import checkpoint_seq

from myvitimm import myvit_base_patch16_384, myvit_small_patch16_384, myvit_tiny_patch16_384, Block, \
    myvit_small_patch16_384, LayerScale
from timm.layers import Mlp

p_s=16
p_n=int(384*384/p_s/p_s)
p_n2=24
def vit_b_480(backbone):
    # vit=myvit_b_p_s(weights='DEFAULT', progress=True)
    if backbone=='small':
        vit=myvit_small_patch16_384(pretrained=True)
    elif backbone=='base':
        vit = myvit_base_patch16_384(pretrained=True)
    elif backbone=='tiny':
        vit = myvit_tiny_patch16_384(pretrained=True)
    else:
        raise ValueError(backbone)
    return vit
class Inter_Frame_Attentionv2(nn.Module):
    def __init__(self,dim, num_heads, mlp_ratio, qkv_bias, attn_drop=0, drop_path=0,
                init_values=0.1,norm_layer=nn.LayerNorm,act_layer=nn.GELU,proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q0 = nn.Linear(dim, dim,bias=qkv_bias)
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2,bias=qkv_bias)

        self.proj1 = nn.Linear(dim, dim)
        self.ln1=nn.LayerNorm(dim)

        self.dim=dim
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,)

    def forward(self,feat):
        x0=torch.cat((feat[:,1,:].unsqueeze(1).clone(),feat[:,0:-1]),dim=1)
        x=feat
        x1=torch.cat((feat[:,1:],feat[:,-2].unsqueeze(1).clone()),dim=1)

        B,T, N, C = x.shape

        qkv = self.kv(x).reshape(B,T, N, 2, self.num_heads, self.head_dim).permute(3, 0,1, 4, 2, 5)
        k, v = qkv.unbind(0)
        # k = self.k(x).reshape(B, T, N, self.num_heads, self.head_dim).transpose(-2, -3)
        # v = self.v(x).reshape(B, T, N, self.num_heads, self.head_dim).transpose(-2, -3)

        q0 = self.q0(x0).reshape(B, T, N, self.num_heads, self.head_dim).transpose(-2, -3)
        attn0 = q0 @ k.transpose(-2, -1)
        q1=self.q1(x1).reshape(B, T, N, self.num_heads, self.head_dim).transpose(-2, -3)
        attn1 = q1 @ k.transpose(-2, -1)

        attn0 = attn0.softmax(dim=-1)
        attn1=attn1.softmax(dim=-1)

        v0=attn0 @ v
        v0=v0.transpose(2, 3).reshape(B,T, N, self.dim)
        # v0 = self.ln1(self.proj1(v0))
        # v0=self.ls1(v0)

        v1 = attn1 @ v
        v1 = v1.transpose(2, 3).reshape(B, T, N, self.dim)
        # v1 = self.ln1(self.proj1(v1))
        # v1 = self.ls1(v1)
        #
        return self.ls2(self.mlp(v1+v0))+feat
def gen_non_neighbor_mask(B,N):
    mask=torch.zeros((B*B,B,B),device='cuda',requires_grad=False)
    for i in range(B):
        for j in range(B):
            x0=0 if i-N<0 else i-N
            y0 = 0 if j - N < 0 else j - N
            x1 = B if i + N > B else i + N+1
            y1 = B if j + N >B else j + N+1
            mask[i*B+j,x0:x1,y0:y1]=1
    mask=mask.view(B*B,B*B)
    return mask
# @torch.jit.script
# @torch.jit.script
def do_maskv1(refs:torch.Tensor,mask:torch.Tensor,mask_all:torch.Tensor):

    B,T ,_, _, _ = refs.shape
    refs2 = torch.zeros((B,T,3, p_n, p_s, p_s),device='cuda')
    for i in range(p_n2):
        for j in range(p_n2):
            refs2[:, :, :,i * p_n2 + j, :] = refs[:, :,:, i * p_s:(i + 1) * p_s, j * p_s:(j + 1) * p_s]
    refs2=refs2.reshape(B*T,3,p_n,p_s,p_s)
    refs2 = refs2.transpose(1, 2)
    B,N,_,_,_=refs2.shape
    ref_vis=refs2[~mask].reshape(B,-1,3,p_s,p_s)
    ref_mask=refs2[mask].reshape(B,-1,3,p_s,p_s)
    refs2=torch.cat((ref_vis,ref_mask),dim=1)
    B, N = mask_all.shape
    refs2 = refs2[mask_all].view(B,-1,3,p_s,p_s)
    return refs2
# @torch.jit.script
def do_maskv2(refs:torch.Tensor,mask:torch.Tensor,mask_all:torch.Tensor):

    B,T ,_, _, _ = refs.shape
    refs2 = torch.zeros((B,T,3, p_n, p_s, p_s),device='cuda')
    for i in range(p_n2):
        for j in range(p_n2):
            refs2[:, :, :,i * p_n2 + j, :] = refs[:, :,:, i * p_s:(i + 1) * p_s, j * p_s:(j + 1) * p_s]
    refs2=refs2[:,2:,:].reshape(B*(T-2),3,p_n,p_s,p_s)
    refs2 = refs2.transpose(1, 2)
    B,N,_,_,_=refs2.shape
    ref_vis=refs2[~mask].reshape(B,-1,3,p_s,p_s)
    ref_mask=refs2[mask].reshape(B,-1,3,p_s,p_s)
    refs2=torch.cat((ref_vis,ref_mask),dim=1)
    B, N = mask_all.shape
    refs2 = refs2[mask_all].view(B,-1,3,p_s,p_s)
    return refs2

def do_maskv3(refs:torch.Tensor,mask:torch.Tensor):

    B,T ,_, _, _ = refs.shape
    refs2 = torch.zeros((B,T,3, p_n, p_s, p_s),device='cuda')
    for i in range(p_n2):
        for j in range(p_n2):
            refs2[:, :, :,i * p_n2 + j, :] = refs[:, :,:, i * p_s:(i + 1) * p_s, j * p_s:(j + 1) * p_s]
    refs2 = refs2.transpose(2, 3)
    B,T,N,_,_,_=refs2.shape
    ref_mask=refs2[mask].reshape(B,T,-1,3,p_s,p_s).view(B*T,-1,3,p_s,p_s)
    return ref_mask

def patchlize(refs:torch.Tensor):

    B,T ,_, _, _ = refs.shape
    refs2 = torch.zeros((B,T,3, p_n, p_s, p_s),device='cuda')
    for i in range(p_n2):
        for j in range(p_n2):
            refs2[:, :, :,i * p_n2 + j, :] = refs[:, :,:, i * p_s:(i + 1) * p_s, j * p_s:(j + 1) * p_s]
    refs2 = refs2.transpose(2, 3)
    B,T,N,_,_,_=refs2.shape
    refs2=refs2.view(B*T,p_n,3,p_s,p_s)
    return refs2
def undo_mask(ref:torch.Tensor,recs:torch.Tensor,masks:torch.Tensor,pixelnorm_):
    B, T, _, _, _ = ref.shape
    recs = recs.view(B, T, -1, 3, 16, 16)
    refss=[]
    for t in range(T):
        refs=ref[0,t,:,:,:]
        rec=recs[0,t,:,:,:]
        mask=masks[0,t]
        f=0
        for i in range(p_n2):
            for j in range(p_n2):
                if mask[i * p_n2 + j]:
                    refs[:, i * p_s:(i + 1) * p_s, j * p_s:(j + 1) * p_s]=rec[f,:]
                    f+=1
        refss.append(pixelnorm_(refs.to('cpu'),True).permute((1, 2, 0)).data.numpy().astype(np.uint8))
        refss.append(np.ones((10,384,3),dtype=np.uint8)*255)
    return refss
# @torch.jit.script
def undo_maskv2(refs:torch.Tensor,rec:torch.Tensor,mask:torch.Tensor,mask_all:torch.Tensor):
    i=0
    refs=refs[i]
    rec=rec[i]
    mask=mask[i]
    mask_all=mask_all[i]
    coords=torch.zeros((p_n2,p_n2,2))
    for i in range(p_n2):
        for j in range(p_n2):
            coords[i,j,0]=i
            coords[i, j, 1] = j
    coords=coords.view(p_n,2)
    coords_vis = coords[~mask].view(-1,2)
    coords_mask = coords[mask].view(-1,2)
    coords=torch.cat((coords_vis,coords_mask),dim=0)
    coords=coords[mask_all].view(-1,2)
    for i in range(coords.shape[0]):
        x=coords[i,0].item()
        y=coords[i,1].item()
        x=int(x)
        y=int(y)
        refs[:, x * p_s:(x + 1) * p_s, y * p_s:(y + 1) * p_s]=rec[i,:]
    return refs
class UnPatch(nn.Module):
    def __init__(self,in_dim=384,blk_sz=p_s):
        super().__init__()
        self.blk_sz=blk_sz
        self.liner=nn.Linear(in_dim,blk_sz*blk_sz*3)

    def forward(self,x):
        x=self.liner(x)#B N b*b*3
        B,N,C=x.shape
        x=x.view(B,N,3,self.blk_sz,self.blk_sz)
        return x


class RadomMask:
    def __init__(self, input_size, mask_ratio,num_patches:int=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size
        if num_patches is None:
            self.num_patches = self.height * self.width
        else:
            self.num_patches=num_patches
        self.num_mask = int(mask_ratio * self.num_patches)

    def __shuffle(self,x:torch.Tensor):
        idx = torch.randperm(x.shape[0])  # 生成一个长度为3的随机序列
        x = x[idx]
        return x

    def __call__(self,B):
        mask = torch.cat((
            torch.zeros(B,self.num_patches - self.num_mask),
            torch.ones(B,self.num_mask),
        ),dim=1)
        for i in range(B):
            mask[i]=self.__shuffle(mask[i])
        return mask.to(torch.bool).to('cuda').detach() # [196]

class STBlock(nn.Module):
    def __init__(self, dim, drop_path=0,
                 num_heads=3, mlp_ratio=4., qkv_bias=False, attn_drop=0.,
                  norm_layer=nn.LayerNorm, init_values=None
                 ):
        super().__init__()
        self.block=Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer,
                init_values=init_values)
        self.inter_frame_attn=Inter_Frame_Attentionv2(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer,
                init_values=init_values)
    def forward(self,x):
        B,T,N,C=x.shape
        x=x.reshape(B*T,N,C)
        x=self.block(x)
        x=x.view(B,T,N,C)
        x=self.inter_frame_attn(x)
        return x

class DecoderST(nn.Module):

    def __init__(self, embed_dim=256, depth=1,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, attn_drop_rate=0.1,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,pos_embed:torch.Tensor=None
                 ):
        super().__init__()
        self.linear=nn.Linear(384,embed_dim)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            STBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        # self.un_patch=nn.ConvTranspose2d(embed_dim, 3, kernel_size=p_s, stride=p_s, bias=True)
        self.un_patch=UnPatch(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, p_n + 1, embed_dim) * .02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1,1, embed_dim))
        self.grad_checkpointing=False
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x,mask):
        # x=F.gelu(self.linear(x))
        B,T,N=mask.shape
        _,_,_,C=x.shape

        expand_pos_embed = self.pos_embed.expand(B, T, -1, -1)
        expand_pos_embed_tk = expand_pos_embed[:, :, 1:]
        pos_emd_vis = expand_pos_embed_tk[~mask].reshape(B, T, -1, C)
        pos_emd_mask = expand_pos_embed_tk[mask].reshape(B, T, -1, C)
        m=int(p_n-mask[0,0,:].sum())
        x = torch.cat((x[:,:, 0].unsqueeze(2) + expand_pos_embed[:,:, 0].unsqueeze(2), x[:, :,1:m+1] + pos_emd_vis,pos_emd_mask+x[:,:,m+1:]+self.mask_token), dim=2)


        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.blocks(x)
        for blk in self.blocks:
            x = blk(x)

        h=x[:,:,0]
        x = x[:, :, 1 + m:]
        x = x.reshape(B, T, -1, C).view(B * T, -1, C)

        # x=x.reshape(B*N,C)
        # BN 3 p_s p_s
        pix=self.un_patch(x)
        # x=x.view((B,N,3,p_s,p_s))
        return pix,h

    def infer(self, feat):  # B,T,N,512
        # feat = F.gelu(self.linear(feat))
        # feat = feat.unsqueeze(0)
        # features.append(feat[0, :, 1:].mean(dim=1).to('cpu'))
        # features.append(feat[0, :, 0].to('cpu'))
        for blk in self.blocks:
            feat = blk(feat)
            # features.append(feat[0, :, 1:].mean(dim=1).to('cpu'))
            # features.append(feat[0, :, 0].to('cpu'))
        # features = torch.cat(features, dim=-1)
        features=feat[:, :, 0]#.to('cpu')
        # features=feat[0, :, 1:].mean(dim=1).to('cpu')
        return features

class Decoder0(nn.Module):

    def __init__(self, in_dim=384, embed_dim=384, depth=1,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,pos_embed:torch.Tensor=None
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.decopose=nn.Linear(in_dim,embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        # self.un_patch=nn.ConvTranspose2d(embed_dim, 3, kernel_size=p_s, stride=p_s, bias=True)
        self.un_patch=UnPatch(embed_dim)
        # self.un_patch=nn.Sequential(nn.Linear(embed_dim,2048),nn.ReLU(True),nn.Linear(2048,p_s*p_s*3))
        self.norm =  norm_layer(in_dim)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x):
        # x=self.decopose(x)

        x=checkpoint_seq(self.blocks,x)
        x=x[:,1:]
        # x=x.reshape(B*N,C)
        # BN 3 p_s p_s
        x=self.un_patch(x)
        # x=x.view((B,N,3,p_s,p_s))
        return x

class DSMAE(nn.Module):
    def __init__(self,rt,depth=12):
        super().__init__()
        self.decoder = DecoderST(embed_dim=384,num_heads=8,depth=1)

        self.backbone = myvit_small_patch16_384(pretrained=True,depth=depth)
        self.dim = 384
        self.backbone.grad_checkpointing=True
        self.masking1=RadomMask(input_size=p_n2,mask_ratio=rt)
        # self.feature_projection=nn.Linear(384,l_dim)
        self.mask_token=nn.Parameter(torch.normal(mean=0.2, std=0.5, size=(1,1,384)))
        self.cls = nn.Sequential(nn.Linear(384 , 256),nn.GELU(),nn.Linear(256,18))
        self.proj=nn.Linear(384,384)

    # @torch.no_grad()
    # def forward(self, dis):
    #     T, _, _, _ = dis.shape
    #     feat = self.backbone.forward_features(dis)
    #     feat = self.decoder.infer(feat)
    #     feat = feat.to('cpu')
    #     # feat = torch.cat((feat[:, 1:].mean(dim=1), feats), dim=-1)
    #     # feat = torch.cat((feat[:, 0], feats), dim=-1)
    #     return feat  # T C


    #Batch infer
    def forward(self, dis):
        B, T, _, _, _ = dis.shape
        dis = dis.view(B * T, 3, 384, 384)
        feat = self.backbone.forward_features(dis)
        feat = feat.view(B, T, -1, 384)
        feat = self.decoder.infer(feat)
        # feat = feat.to('cpu')
        # feat = torch.cat((feat[:, 1:].mean(dim=1), feats), dim=-1)
        # feat = torch.cat((feat[:, 0], feats), dim=-1)
        return feat  # T C


    def lwf2(self,x,feats):
        B, T, _, _, _ = x.shape
        x = x.view(B * T, 3, 384, 384)
        feats_=self.backbone.forward_features(x)
        feats_=self.proj(feats_)
        return F.l1_loss(feats_,feats)
    def mae(self,ref,dis):
        B,T,_, _, _ = dis.shape
        mask1 = self.masking1(B*T)
        dis=dis.view(B*T,3,384,384)
        ref=ref.view(B*T,3,384,384)
        feat = self.backbone.forward_features_maskv3(dis, ref, mask1)
        feat=feat.view(B,T,-1,384)

        mask1=mask1.view(B,T,-1)

        y, h = self.decoder(feat, mask1)
        # h=torch.cat((feat[:,:,0],h),dim=-1)
        pre = self.cls(h.mean(dim=1))
        return y, mask1, pre

    def warm_up(self,ref,dis):
        with torch.no_grad():
            B, T, _, _, _ = dis.shape
            mask1 = self.masking1(B * T)
            dis = dis.view(B * T, 3, 384, 384)
            ref = ref.view(B * T, 3, 384, 384)
            feat = self.backbone.forward_features_maskv3(dis, ref, mask1)
            feat = feat.view(B, T, -1, 384)

            mask1 = mask1.view(B, T, -1)
        y, h = self.decoder(feat, mask1)
        # h = torch.cat((feat[:, :, 0], h), dim=-1)
        pre = self.cls(h.mean(dim=1))
        return y, mask1, pre


class Nomalize:
    def __init__(self,device='cuda'):
        self.defaultmean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device, dtype=torch.float,
                                        requires_grad=False).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        self.defaultstd = torch.tensor(IMAGENET_DEFAULT_STD, device=device, dtype=torch.float,
                                       requires_grad=False).unsqueeze(
            -1).unsqueeze(-1).unsqueeze(0)


    def __call__(self, x,denorm:bool=False):
        if denorm:
            x.mul_(self.defaultstd.squeeze(0)).add_(self.defaultmean.squeeze(0)).mul_(255)
        else:
            x.div_(255.0).sub_(self.defaultmean).div_(self.defaultstd)
        return x