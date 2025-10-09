import os
import numpy as np
import pandas as pd
import json
from RegGAN.reg_gan_dataloader import MyDataLoader
from RegGAN.trainer.reg_gan_trainer import RegGANTrainer
import datetime

traindir = r"/media/dataset01"
spath = r'/media/dataset01'


#config
cfg = {
    'df_path': os.path.join(spath, 'select.ftr'),
    'cts':('CT1','CT2'),
    'img_shape':(256,256,16), #(y,x,z)
    'window1':[(500,50)], 
    'window2':[(500,50)],
    'lrs':(0.0001, 0.1), 
    'L_weights':(1,100),
    'grid':(4,4,4),
    'regist':True, #regist=False enables Pix2Pix mode (direct NCCT-to-vCECT without registration).
    'randomshift':0.1
}

now = datetime.datetime.now() 
df0 = pd.read_feather(cfg['df_path']) 


#256×256
df0.Rows=256 #y
df0.Columns=256 #x
DL = MyDataLoader(df0, cts=cfg['cts'], img_shape=cfg['img_shape'],grid=cfg['grid'],window1=cfg['window1'], window2=cfg['window2'])

for i in range(len(DL.total_samples[0])):
 DL.total_samples[0][i][1]=128 #x
 DL.total_samples[0][i][2]=128 #y


gan=RegGANTrainer(cfg, savepath = traindir + '/Models' + now.strftime("%Y%m%d%H%M")  + '/', data_loader=DL)

gan.train(epochs=15, start_epochs=0, batch_size=1, sample_interval=100, model_interval=1, plot_interval=1)
