# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import copy
import torch
from torch_utils import nethook
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

from .loss import Loss
from .networks_stylegan2 import Conv2dLayer

#----------------------------------------------------------------------------
# hyperparams (hard-coded)
kl_wt = 1000
highp = 1
subspace_freq = 4
feat_ind = 3
feat_const_batch = 2
n_sample = 4
patch_size = 4
feat_res = 128
subspace_std = 0.05
n_train = 10
lowp, highp = 0, 1


class FewShotAdaptLoss(Loss):
    def __init__(self, device, G, D, F, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

        # my stuff
        self.F = F
        self.init_z = torch.randn(n_train, self.G.z_dim, device=self.device)
        self.oG = copy.deepcopy(G).requires_grad_(False).to('cuda:1')
        self.all_g_layers = [name for name, mod in self.oG.named_modules() if '.L' in name and '.affine' not in name]
        self.all_d_layers = [name for name, mod in self.D.named_modules() if 'b32.conv' in name or 'b16.conv' in name]
        self.sfm = torch.nn.Softmax(dim=1)
        self.kl_loss = torch.nn.KLDivLoss()
        self.sim = torch.nn.CosineSimilarity()


    def run_G(self, z, c, batch_idx, update_emas=False):
        sample_anchor = batch_idx % subspace_freq == 0 # defines whether we sample from anchor region in this iteration or other

        if sample_anchor:
            z = get_subspace(self.init_z.clone())
            ws = self.G.mapping(z, c, update_emas=update_emas)
        else:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, batch_idx, blur_sigma=0, update_emas=False):
        sample_anchor = batch_idx % subspace_freq == 0 # defines whether we sample from anchor region in this iteration or other

        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        
        if sample_anchor:
            logits = self.D(img, c, update_emas=update_emas)
        else:
            p_ind = np.random.randint(lowp, highp)
            with nethook.Trace(self.D, self.all_d_layers[p_ind], stop=True) as ret:
                self.D(img, c, update_emas=update_emas)
            feat = ret.output
            logits = self.F(feat, p_ind)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, batch_idx):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            self.F.requires_grad_(False)
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, batch_idx)
                gen_logits = self.run_D(gen_img, gen_c, batch_idx, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                loss_Grel = self.distance_consistency()
                training_stats.report('Loss/G/loss_rel', loss_Grel)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Grel).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            self.F.requires_grad_(False)
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], batch_idx)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            self.F.requires_grad_(True)
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, batch_idx, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, batch_idx, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            self.F.requires_grad_(True)
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, batch_idx, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()


    def distance_consistency(self):
        with torch.set_grad_enabled(False):
            z = torch.randn(feat_const_batch, self.G.z_dim, device='cuda:1')
            feat_ind = np.random.randint(1, self.G.num_ws - 2, size=feat_const_batch)

            # computing source distances
            with nethook.TraceDict(self.oG, layers=self.all_g_layers) as ret:
                self.oG(z, None)
                feat_source = [ret[k].output for k in self.all_g_layers]

            dist_source = torch.zeros([feat_const_batch, feat_const_batch - 1]).to('cuda:1')

            # iterating over different elements in the batch
            for pair1 in range(feat_const_batch):
                tmpc = 0
                # comparing the possible pairs
                for pair2 in range(feat_const_batch):
                    if pair1 != pair2:
                        anchor_feat = torch.unsqueeze(
                            feat_source[feat_ind[pair1]][pair1].reshape(-1), 0)
                        compare_feat = torch.unsqueeze(
                            feat_source[feat_ind[pair1]][pair2].reshape(-1), 0)
                        dist_source[pair1, tmpc] = self.sim(
                            anchor_feat, compare_feat)
                        tmpc += 1
            dist_source = self.sfm(dist_source).to(self.device)

        # computing distances among target generations
        with nethook.TraceDict(self.G, layers=self.all_g_layers) as ret:
            z = z.to(self.device)
            self.G(z, None)
            feat_target = [ret[k].output for k in self.all_g_layers]

        dist_target = torch.zeros([feat_const_batch, feat_const_batch - 1]).to(self.device)

        # iterating over different elements in the batch
        for pair1 in range(feat_const_batch):
            tmpc = 0
            for pair2 in range(feat_const_batch):  # comparing the possible pairs
                if pair1 != pair2:
                    anchor_feat = torch.unsqueeze(
                        feat_target[feat_ind[pair1]][pair1].reshape(-1), 0)
                    compare_feat = torch.unsqueeze(
                        feat_target[feat_ind[pair1]][pair2].reshape(-1), 0)
                    dist_target[pair1, tmpc] = self.sim(anchor_feat, compare_feat)
                    tmpc += 1
        dist_target = self.sfm(dist_target)
        rel_loss = kl_wt * self.kl_loss(torch.log(dist_target), dist_source) # distance consistency loss 
        return rel_loss

#----------------------------------------------------------------------------
def get_subspace(init_z):
    std = subspace_std
    bs = n_sample
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    return z


class Extra(torch.nn.Module):
    # to apply the patch-level adversarial loss, we take the intermediate discriminator feature maps of size [N x N x D], and convert them into [N x N x 1]

    def __init__(self):
        super().__init__()

        self.new_conv = torch.nn.ModuleList()
        self.new_conv.append(Conv2dLayer(512, 1, 3))
        self.new_conv.append(Conv2dLayer(512, 1, 3))

    def forward(self, inp, ind):
        out = self.new_conv[ind](inp)
        return out
