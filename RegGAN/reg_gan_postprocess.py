#!/usr/bin/env python3
# reg_gan_torch_clean.py
# Cleaned for: regist always True; 2D/3D auto-switch; minimal model choices; loop_over_case kept.
# Author: Refactored from reg_gan_torch.py

import os
import glob
import csv
from io import StringIO
import concurrent.futures
import uuid
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted

import torch
import pydicom

# Utility
from RegGAN.reg_gan_dataloader import WND, rWND
from RegGAN.reg_gan_metrics import * 

def dcmcjpeg(input_path, output_path):
    cmd = ["dcmcjpeg", "+sr", input_path, output_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(f"Compressed: {input_path}")
    else:
        print(f"Failed: {input_path}")


def compress_all_dicoms(dicom_folder):
    dicoms = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.lower().endswith(".dcm")]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(dcmcjpeg, dcm, dcm) for dcm in dicoms]
        concurrent.futures.wait(futures)


def generate_dicom_uid():
    uid_base = "1.2.826.0.1.3680043.2.1125"
    rand_uid = uuid.uuid4().int >> 64 
    return f"{uid_base}.{rand_uid}"

def center_crop(img, target_h, target_w):
    h, w, d = img.shape
    start_y = (h - target_h) // 2
    start_x = (w - target_w) // 2
    return img[start_y:start_y+target_h, start_x:start_x+target_w, :]



def make_directory(spath, dirname):
    dirpath = os.path.join(spath, dirname)
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    return dirpath


def my_dicoms_to_dataframe(basedir, cts):
    caselist = [os.path.join(basedir, x) for x in os.listdir(basedir) if 'case' in x]
    file_list = []
    for x in cts:
        file_list.extend(glob.glob(os.path.join(basedir, '*', x, '*.dcm')))
        file_list.extend(glob.glob(os.path.join(basedir, '*', x, '*.DCM')))

    tdcmpath = os.path.join(caselist[0], cts[0])
    tdcmpath = [os.path.join(tdcmpath, x) for x in os.listdir(tdcmpath) if x.lower().endswith('.dcm')][0]
    tdcm = pydicom.dcmread(tdcmpath)

    headers = ['filepath']
    for x in tdcm:
        if 'Image Position (Patient)' in x.name or 'Rows' in x.name or 'Columns' in x.name or 'Rescale Intercept' in x.name or 'Rescale Slope' in x.name:
            name = x.name.replace(' ', '')
            headers.append(name)

    output = StringIO()
    csv_writer = csv.DictWriter(output, fieldnames=headers)
    csv_writer.writeheader()

    for f in tqdm(file_list):
        file = pydicom.dcmread(f)
        row = {'filepath': f}
        for x in file:
            if 'Image Position (Patient)' in x.name or 'Rows' in x.name or 'Columns' in x.name or 'Rescale Intercept' in x.name or 'Rescale Slope' in x.name:
                name = x.name.replace(' ', '')
                row[name] = x.value
        unwanted = set(row) - set(headers)
        for unwanted_key in unwanted:
            del row[unwanted_key]
        csv_writer.writerow(row)

    output.seek(0)
    df = pd.read_csv(output)
    df['pid'] = df['filepath'].apply(lambda x: x.split(os.sep)[-3])
    df['ct'] = df['filepath'].apply(lambda x: x.split(os.sep)[-2])
    df['zpos'] = df['ImagePosition(Patient)'].apply(lambda x: x.split()[-1]).str.strip(']')

    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    df.to_feather(os.path.join(basedir, 'headers.ftr'))
    return df


def loop_over_case(gan, model, case, savedir, savefolder="prediction", notruth=False, opposite=False, compress=False, metrics=([500,50],[500,50]), saveimage=True, SeriesNumber="100", SeriesDescription="fake"):
    pid, zs = case 
    #dcm_A:CT2,dcm_B:CT1
    dcm_A, dcm_B = gan.data_loader.load_dicoms(pid, (0,zs+1))

    #if notruth:
    if len(dcm_A)==0:
        dcm_A = np.zeros(dcm_B.shape, dtype=dcm_B.dtype)

    if opposite:
      dcm_A, dcm_B = dcm_B, dcm_A


    if dcm_A.shape[0] != dcm_B.shape[0] or dcm_A.shape[1] != dcm_B.shape[1]:
        target_h = min(dcm_A.shape[0], dcm_B.shape[0])
        target_w = min(dcm_A.shape[1], dcm_B.shape[1])

        if dcm_A.shape[0] != target_h or dcm_A.shape[1] != target_w:
            dcm_A = center_crop(dcm_A, target_h, target_w)
        if dcm_B.shape[0] != target_h or dcm_B.shape[1] != target_w:
            dcm_B = center_crop(dcm_B, target_h, target_w)


    xm, ym = 0, 0 
    if dcm_A.shape[0]!=gan.data_loader.img_shape[0] or dcm_A.shape[1]!=gan.data_loader.img_shape[1]:
      target_h = gan.data_loader.img_shape[0]
      target_w = gan.data_loader.img_shape[1]
      dcm_A = center_crop(dcm_A, target_h, target_w)
      dcm_B = center_crop(dcm_B, target_h, target_w)

    
    a = []
    b = []
    for w in gan.data_loader.window2:
        a.append(WND(dcm_A,w)) 
    for w in gan.data_loader.window1:
        b.append(WND(dcm_B,w))
    tot_A = np.stack(a, axis=-1) #tot_A.shapeは(512, 512, zs, 3)
    tot_B = np.stack(b, axis=-1) #tot_B.shapeは(512, 512, zs, 3)
    tot_A = tot_A.astype('float32')/127.5 - 1. #convert to -1 to 1
    tot_B = tot_B.astype('float32')/127.5 - 1. #convert to -1 to 1
    
    fakes_raw = np.full((gan.data_loader.img_shape[0],gan.data_loader.img_shape[1],zs),0,dtype=tot_B.dtype)
    counts_raw = np.full((gan.data_loader.img_shape[0],gan.data_loader.img_shape[1],zs),0,dtype=int)
    
    for i in tqdm(range(zs+1-gan.data_loader.img_shape[2])): 
        imgs_B = np.expand_dims(tot_B[:,:,i:i+gan.data_loader.img_shape[2],:], axis=0) 
        imgs_B = np.transpose(imgs_B, (0, 4, 1, 2, 3)) #shape[1,3,512,512,z] 
        if gan.data_loader.img_shape[2]==1: #2dの時
           imgs_B = imgs_B[:,:,:,:,0] #shape[1,3,512,512]
        imgs_B = torch.from_numpy(imgs_B.astype(np.float32)).clone().cuda() 
        
        imgs_A = np.expand_dims(tot_A[:,:,i:i+gan.data_loader.img_shape[2],:], axis=0) 
        imgs_A = np.transpose(imgs_A, (0, 4, 1, 2, 3)) #shape[1,3,512,512,z] 
        if gan.data_loader.img_shape[2]==1: #2dの時
           imgs_A = imgs_A[:,:,:,:,0] #shape[1,3,512,512]
        imgs_A = torch.from_numpy(imgs_A.astype(np.float32)).clone().cuda() 


        if model: 
          fake_A = model(imgs_B)  
        else:
          fake_A = imgs_B #generator not used

        fake_A = fake_A.to('cpu').detach().numpy().copy() 
        if gan.data_loader.img_shape[2]==1: #2D input
          fake_A = np.transpose(fake_A, (0,2,3,1)) #shape[1,512,512,3]
        else: #3D input
          fake_A = np.transpose(fake_A, (0,2,3,4,1)) #[1,512,512,z,3]
       
        fake_A = 0.5 * fake_A + 0.5 #convert to 0-1
        if gan.data_loader.img_shape[2]==1: #2D input
                    fake_A = np.expand_dims(fake_A,axis=3) #fake_A.shapeは(1, 512, 512, 1, 3)
        fake_A = rWND(255.*fake_A[:,:,:,:,0], gan.data_loader.window2[0]) #Convert to HU
    
        fakes_raw[:,:,i:i+gan.data_loader.img_shape[2]] += fake_A[0] 
        counts_raw[:,:,i:i+gan.data_loader.img_shape[2]] += 1 
    
    mcounts = counts_raw.copy() 
    mcounts[mcounts==0] = 1 
    fakes = np.divide(fakes_raw, mcounts) 
    
    
    df1 = gan.data_loader.df
    dcms1 = df1[(df1['pid']==pid)&(df1['ct']==gan.data_loader.cts[0])]['filepath'].tolist() 
    dcms1 = natsorted(dcms1)

    if saveimage: 
      newpath = os.path.join(savedir, pid, savefolder)
      os.makedirs(newpath, exist_ok=True)

    else:
      newpath = None  # or skip all saving operations entirely

    study_uid = generate_dicom_uid()
    series_uid = generate_dicom_uid()

    for N, y in tqdm(enumerate(dcms1)):
        x = fakes[:,:,N]

        ds = pydicom.dcmread(y)
        ds.decompress()
    
        x = (x-float(ds.RescaleIntercept))/float(ds.RescaleSlope)
    
        x = x.astype('int16')

        ds.Rows=gan.data_loader.img_shape[0]
        ds.Columns=gan.data_loader.img_shape[1]
    
        ds.PixelData = x.tobytes()

        # ★Adjust ImagePositionPatient
        pixel_spacing_x = float(ds.PixelSpacing[0])
        pixel_spacing_y = float(ds.PixelSpacing[1])
        ds.ImagePositionPatient[0] += xm * pixel_spacing_x
        ds.ImagePositionPatient[1] += ym * pixel_spacing_y

        ds.PatientID = 'Anonymous'
        ds.SeriesNumber = SeriesNumber 
        ds.SeriesDescription = SeriesDescription 
        ds.StudyDescription = SeriesDescription 
        ds.StudyID = SeriesDescription 
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = generate_dicom_uid()
   
        if saveimage: 
          newfile = os.path.join(newpath, os.path.basename(y))
          ds.save_as(newfile) #Save DICOM

    if saveimage and compress:
      compress_all_dicoms(newpath)


    if metrics:
        dcm_A = WND(dcm_A, metrics[0]) #convert original 3D images to mediastinal window (L:350,C:40) #true
        fakes = WND(fakes, metrics[1]) #convert generated 3D images to mediastinal window (L:350,C:40) #fake
        #calculate metrics
        ssim = SSIM(dcm_A, fakes)
        mae = MAE(dcm_A, fakes)
        psnr = PSNR_slice_mean(dcm_A, fakes)
        lpips = LPIPS(dcm_A, fakes)

        #save metrics as csv file
        outcsv = os.path.join(savedir, "metrics.csv") #save csv
        if not os.path.exists(outcsv):
          with open(outcsv, "w", newline='') as f:
            writer=csv.writer(f)
            writer.writerow(["Case", "ssim", "mae", "psnr", "lpips"])
            writer.writerow([pid, ssim, mae, psnr, lpips])
        else:  
          with open(outcsv, "a", newline='') as f:
            writer=csv.writer(f)
            writer.writerow([pid, ssim, mae, psnr, lpips])

