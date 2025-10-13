import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from RegGAN.Model.reg_gan_Models_torch import *
from RegGAN.trainer.reg_gan_trainer import RegGANTrainer
from RegGAN.reg_gan_postprocess import loop_over_case, make_directory, my_dicoms_to_dataframe
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from RegGAN.reg_gan_dataloader import MyDataLoader


# --- Directories ---
testdir = r'/media/dataset01'        # Directory containing test DICOMs
spath = r'/media/dataset01'          # Directory containing cfg.json
model_folder = os.path.join('/media/model_folder') # Directory containing model files

# --- Load configuration generated during training ---
with open(os.path.join(spath, 'cfg.json')) as json_file:
    cfg = json.load(json_file)


# --- Load case list (Feather format) ---
df0 = pd.read_feather(os.path.join(testdir, 'select.ftr'))


# --- Set image size to 256×256 for inference ---
df0.Rows = 256  # height (y-axis)
df0.Columns = 256  # width (x-axis)

DL = MyDataLoader(
    df0,
    cts=cfg['cts'],
    img_shape=cfg['img_shape'],
    grid=cfg['grid'],
    window1=cfg['window1'],
    window2=cfg['window2'],
    rescale_intensity=cfg['rescale_intensity'],
    splitvar=1
)

# Fix the central crop position
for i in range(len(DL.total_samples[0])):
    DL.total_samples[0][i][1] = 128  # x-center
    DL.total_samples[0][i][2] = 128  # y-center


# --- Initialize RegGAN Trainer (in inference mode) ---
gan = RegGANTrainer(cfg, savepath=testdir, data_loader=DL, opposite=False)


# --- Load pretrained model weights ---
directory = model_folder + '/Reg_GAN.pth'  # path to the trained model
gan.netG_A2B.load_state_dict(torch.load(directory)['G_state_dict'])
gan.netG_A2B.eval()


# --- Load test case list ---
L = gan.data_loader.case_split[0]  # Independent test dataset


# --- Set output directory ---
savedir = make_directory(model_folder, 'results')  # generated results will be saved here


# --- Generate virtual CECT images ---
choice = np.arange(len(L))  # list of case indices
samples = []  # for storing results (optional visualization)

for case in tqdm(choice):
    samples.append(
        loop_over_case(
            gan,
            model=gan.netG_A2B,
            case=L[case],
            savedir=savedir,
            notruth=True,        # set True if true CT2 (CECT) is unavailable
            opposite=False,       # set False for normal direction (CT1→CT2)
            compress=True,        # compress DICOMs for storage efficiency
            metrics=([500, 50], [500, 50]),  # evaluation window levels
            saveimage=True        # save output images
        )
    )


