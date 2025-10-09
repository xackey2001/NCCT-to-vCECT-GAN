#!/usr/bin/env python3
# reg_gan_torch_clean.py
# Cleaned for: regist always True; 2D/3D auto-switch; minimal model choices; loop_over_case kept.
# Author: Refactored from reg_gan_torch.py

import os
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# RegGAN modules (actually used)
from RegGAN.Model.reg_gan_Models_torch import (
    Generator_2d, Generator_3d, Discriminator_2d, Discriminator_3d
)
from RegGAN.trainer.reg_gan_layers_unified import smooothing_loss as smoothing_loss_nd
from RegGAN.trainer.reg_gan_transformer_unified import TransformerND
from RegGAN.trainer.reg_gan_reg_unified import Reg



class RegGANTrainer:
    """
    Trainer class. Supports 2D/3D switching and both Reg-GAN and Pix2Pix modes.
    """
    def __init__(self, cfg, savepath, data_loader, opposite=False):
        super().__init__()
        self.cfg = cfg
        self.savepath = savepath
        self.data_loader = data_loader
        self.opposite = opposite
        self.now = datetime.datetime.now()
        self.randomshift = self.cfg.get('randomshift', 0.0)

        # --- Choose 2D vs 3D ---
        self.is_2d = (len(self.cfg["img_shape"]) == 2 or self.cfg["img_shape"][-1] == 1)

        # --- Build Generator ---
        in_ch = len(self.cfg["window1"])
        out_ch = len(self.cfg["window1"])
        if self.is_2d:
            self.netG_A2B = Generator_2d(in_ch, out_ch).cuda()
        else:
            self.netG_A2B = Generator_3d(in_ch, out_ch).cuda()

        # --- Build Discriminator ---
        # regist=True → input = fake_A (in_ch)
        # regist=False → input = concat(imgs_B, fake_A) (in_ch*2)
        if self.is_2d:
            d_in_ch = in_ch if self.cfg.get('regist', True) else in_ch * 2
            self.netD_B = Discriminator_2d(d_in_ch).cuda()
        else:
            d_in_ch = in_ch if self.cfg.get('regist', True) else in_ch * 2
            self.netD_B = Discriminator_3d(d_in_ch).cuda()

        # --- Build Registration (only if regist=True) ---
        if self.cfg.get('regist', True):
            if self.is_2d:
                self.R_A = Reg(
                    dim=2,
                    height=self.cfg["img_shape"][0],
                    width=self.cfg["img_shape"][1],
                    in_channels_a=in_ch,
                    in_channels_b=out_ch
                ).cuda()
            else:
                self.R_A = Reg(
                    dim=3,
                    height=self.cfg["img_shape"][0],
                    width=self.cfg["img_shape"][1],
                    depth=self.cfg["img_shape"][2],
                    in_channels_a=in_ch,
                    in_channels_b=out_ch
                ).cuda()

            self.spatial_transform = TransformerND().cuda()
            self.smoothing_loss = smoothing_loss_nd

        # --- Optimizers ---
        lr = self.cfg["lrs"][0]
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=lr, betas=(0.5, 0.999))
        if self.cfg.get('regist', True):
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=lr, betas=(0.5, 0.999))

        # --- Losses ---
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # --- GAN targets ---
        Tensor = torch.cuda.FloatTensor
        self.target_real = Tensor(1, 1).fill_(1.0)
        self.target_fake = Tensor(1, 1).fill_(0.0)


    def train(self, epochs, start_epochs=0, batch_size=1, sample_interval=100, model_interval=1, plot_interval=1):
        start_time = datetime.datetime.now()
        os.makedirs(self.savepath, exist_ok=True)

        for epoch in range(start_epochs, epochs):
            if not self.data_loader:
                continue

            # ensure log file exists
            open(os.path.join(self.savepath, 'log.txt'), 'a').close()

            for i, (imgs_A, imgs_B, caseNum) in enumerate(self.data_loader.load_batch(batch_size=1)):
                print(str(caseNum))

                # --- random shift augmentation (applied equally to A/B) ---
                non = lambda s: s if s < 0 else None
                mom = lambda s: max(0, s)

                shift_A = np.full(imgs_A.shape, -1.0, dtype=imgs_A.dtype)
                shift_B = np.full(imgs_B.shape, -1.0, dtype=imgs_B.dtype)
                sx = int(self.randomshift * shift_A.shape[2])
                sy = int(self.randomshift * shift_A.shape[1])
                for j in range(shift_A.shape[0]):
                    ox = np.random.randint(2 * sx + 1) - sx
                    oy = np.random.randint(2 * sy + 1) - sy
                    shift_A[j, mom(oy):non(oy), mom(ox):non(ox), :, :] = imgs_A[j, mom(-oy):non(-oy), mom(-ox):non(-ox), :, :]
                    shift_B[j, mom(oy):non(oy), mom(ox):non(ox), :, :] = imgs_B[j, mom(-oy):non(-oy), mom(-ox):non(-ox), :, :]
                imgs_A, imgs_B = shift_A, shift_B

                # --- NHWDC -> NCDHW (or NCHW for 2D) ---
                imgs_A = np.transpose(imgs_A, (0, 4, 1, 2, 3))
                imgs_B = np.transpose(imgs_B, (0, 4, 1, 2, 3))
                if self.is_2d:
                    imgs_A, imgs_B = imgs_A[:, :, :, :, 0], imgs_B[:, :, :, :, 0]

                imgs_A = torch.from_numpy(imgs_A.astype(np.float32)).cuda()
                imgs_B = torch.from_numpy(imgs_B.astype(np.float32)).cuda()

                if self.opposite:
                    imgs_A, imgs_B = imgs_B, imgs_A

                # --- weighting (fixed as in original) ---
                Adv_lamda, Corr_lamda, Smooth_lamda = 1.0, 20.0, 10.0


                # --- Train G + Reg ---
                if self.cfg.get('regist', True): #Reg-gan
                  self.optimizer_G.zero_grad()
                  self.optimizer_R_A.zero_grad()

                  fake_A = self.netG_A2B(imgs_B)                         # synthesize
                  Trans = self.R_A(fake_A, imgs_A)                        # flow field
                  SysRegist_fake_A = self.spatial_transform(fake_A, Trans)  # warp synth to align A

                  SR_loss = Corr_lamda * self.L1_loss(SysRegist_fake_A, imgs_A)
                  SM_loss = Smooth_lamda * self.smoothing_loss(Trans)

                  pred_fake0 = self.netD_B(fake_A)
                  adv_loss = Adv_lamda * self.MSE_loss(pred_fake0, self.target_real)

                  total_loss = SM_loss + adv_loss + SR_loss
                  total_loss.backward()
                  self.optimizer_R_A.step()
                  self.optimizer_G.step()


                else:
                # Pix2Pix mode, Train generator
                  self.optimizer_G.zero_grad()
                  fake_A = self.netG_A2B(imgs_B)
                  loss_L1 = self.L1_loss(fake_A, imgs_A) * self.cfg['L_weights'][1]
                  # gan loss: 
                  fake_AB = torch.cat((imgs_B, fake_A), 1)
                  pred_fake = self.netD_B(fake_AB) 
                  loss_GAN_A2B = self.MSE_loss(pred_fake, self.target_real) * self.cfg['L_weights'][0] #self.cfg['L_weights'][0]=1 #デフォルトはself.MSE_loss

                  #total loss
                  total_loss = loss_L1 + loss_GAN_A2B
                  total_loss.backward()
                  self.optimizer_G.step()


                # --- Train D ---
                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_A = self.netG_A2B(imgs_B)

                if self.cfg.get('regist', True): #Reg-GAN
                  pred_fake0 = self.netD_B(fake_A)
                  pred_real = self.netD_B(imgs_A)

                else: #pix2pix
                  pred_fake0 = self.netD_B(torch.cat((imgs_B, fake_A), 1)) #Discriminatorにfake画像を入力して判定させる
                  pred_real = self.netD_B(torch.cat((imgs_B, imgs_A), 1)) #Discriminatorにreal画像を入力して判定させる

                loss_fake = self.MSE_loss(pred_fake0, self.target_fake)
                loss_real = self.MSE_loss(pred_real, self.target_real)
                loss_D_B = loss_fake + loss_real
                loss_D_B.backward()
                self.optimizer_D_B.step()

                elapsed_time = datetime.datetime.now() - start_time


                # --- Logging ---
                if self.cfg.get('regist', True): #Reg-GAN
                  newlog = (
                      "[Epoch %d/%d] [Batch %d/%d] [D loss: %.6f, D fake: %.6f] "
                      "[total loss: %.6f, SM loss: %.6f, adv loss: %.6f, SR loss: %.6f] "
                      "CaseNo: %s time: %s"
                      % (
                          epoch + 1, epochs, i + 1, len(self.data_loader.get_total_samples()[0]),
                          loss_D_B.item(), loss_fake.item(), total_loss.item(), SM_loss.item(), adv_loss.item(), SR_loss.item(),
                          caseNum, elapsed_time
                      )
                  )


                else: #Pix2pix
                  newlog = "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D fake: %f] [total loss: %f, L1_loss: %f, gan_loss: %f] CaseNo: %s time: %s" % (
                      epoch+1, epochs, i+1, len(self.data_loader.get_total_samples()[0]),
                      loss_D_B, loss_fake, total_loss, loss_L1, loss_GAN_A2B,
                      caseNum,
                      elapsed_time,
                  )  


                if (i + 1) % plot_interval == 0:
                    print(newlog)
                    with open(os.path.join(self.savepath, 'log.txt'), 'a') as f:
                        f.write(newlog + '\n')

                if (i + 1) % sample_interval == 0:
                    self.sample_images(epoch, i + 1)


            # --- Save checkpoints ---
            if (epoch + 1) % model_interval == 0:
                if self.cfg.get('regist', True):
                    # Reg-GAN
                    torch.save({
                        'G_state_dict': self.netG_A2B.state_dict(),
                        'D_state_dict': self.netD_B.state_dict(),
                        'R_state_dict': self.R_A.state_dict(),
                        'G_optimizer_state_dict': self.optimizer_G.state_dict(),
                        'D_optimizer_state_dict': self.optimizer_D_B.state_dict(),
                        'R_optimizer_state_dict': self.optimizer_R_A.state_dict()
                    }, os.path.join(self.savepath, f"model_regist_{epoch+1}epochs.pth"))
                else:
                    # Pix2Pix
                    torch.save({
                        'G_state_dict': self.netG_A2B.state_dict(),
                        'D_state_dict': self.netD_B.state_dict(),
                        'G_optimizer_state_dict': self.optimizer_G.state_dict(),
                        'D_optimizer_state_dict': self.optimizer_D_B.state_dict()
                    }, os.path.join(self.savepath, f"model_nonregist_{epoch+1}epochs.pth"))



    # --- Sampling utils (both 2D/3D, regist=True) ---
    def sample_images(self, epoch, i):
        r, c = 1, 3
        imgs_A, imgs_B = self.load_data(batch_size=1)
        imgs_A = np.transpose(imgs_A, (0, 4, 1, 2, 3))
        imgs_B = np.transpose(imgs_B, (0, 4, 1, 2, 3))
        if self.is_2d:
            imgs_A, imgs_B = imgs_A[:, :, :, :, 0], imgs_B[:, :, :, :, 0]

        imgs_A = torch.from_numpy(imgs_A.astype(np.float32)).cuda()
        imgs_B = torch.from_numpy(imgs_B.astype(np.float32)).cuda()

        if self.opposite:
            imgs_A, imgs_B = imgs_B, imgs_A

        fake_A = self.netG_A2B(imgs_B)

        # pick mediastinal channel (-1) and central slice [..,0] for 3D
        if self.is_2d:
            gen_imgs = torch.cat([imgs_B[:, -1, :, :], fake_A[:, -1, :, :], imgs_A[:, -1, :, :]])
        else:
            gen_imgs = torch.cat([imgs_B[:, -1, :, :, 0], fake_A[:, -1, :, :, 0], imgs_A[:, -1, :, :, 0]])

        gen_imgs = gen_imgs.detach().cpu().numpy()
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Fake', 'Real']
        plt.style.use('default')
        fig, axs = plt.subplots(r, c, figsize=(6.4, 3.5))
        for k in range(c):
            axs[k].imshow(gen_imgs[k], cmap='gray', vmin=0, vmax=1)
            axs[k].set_title(titles[k])
            axs[k].axis('off')

        samplepath = os.path.join(self.savepath, self.now.strftime("%Y%m%d%H%M"))
        os.makedirs(samplepath, exist_ok=True)
        fig.savefig(os.path.join(samplepath, f"{epoch}_{i}.png"))
        plt.close()
        print("Sample image saved to ", samplepath)

