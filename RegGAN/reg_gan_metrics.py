import os
import numpy as np
import pydicom
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
import pandas as pd
from natsort import natsorted


def dicom_array(root_dir, HU=True):
    # Extract all .dcm files in the given root directory
    dcms = []
    for d, s, fl in os.walk(root_dir):
        for fn in fl:
            if fn.lower().endswith(".dcm"):
                dcms.append(os.path.join(d, fn))

    # Sort filenames in natural order
    dcms = natsorted(dcms)

    # Use the first file as a reference to allocate the array
    ref_dicom = pydicom.read_file(dcms[0])
    d_array = np.zeros(
        (ref_dicom.Rows, ref_dicom.Columns, len(dcms)),
        dtype=ref_dicom.pixel_array.dtype
    )

    # Read each slice in the correct order
    for i, dcm_path in enumerate(dcms):
        d = pydicom.read_file(dcm_path)
        d_array[:, :, i] = d.pixel_array

    # Convert to HU (Hounsfield Units)
    if HU:
        d_array = d_array * ref_dicom.RescaleSlope + ref_dicom.RescaleIntercept

    return d_array  # shape = (512, 512, number of slices)


def WND(X, W):
    """WND converts CT values (=X) to 0–255 grayscale images based on the CT window (=W)."""
    R = 255. * (X - W[1] + 0.5 * W[0]) / W[0]
    R[R < 0] = 0
    R[R > 255] = 255
    return R


def SSIM(ori_img, gene_img):
    """Compute the mean multi-scale SSIM between two 3D image volumes (slice-wise)."""
    total_ssim = 0
    for i in range(ori_img.shape[2]):
        ori_slice = tf.convert_to_tensor(ori_img[:, :, i:i+1], dtype=tf.float32)  # shape (H, W, 1)
        gene_slice = tf.convert_to_tensor(gene_img[:, :, i:i+1], dtype=tf.float32)
        ori_slice = tf.expand_dims(ori_slice, axis=0)  # shape (1, H, W, 1)
        gene_slice = tf.expand_dims(gene_slice, axis=0)
        ssim_slice = tf.image.ssim_multiscale(ori_slice, gene_slice, max_val=255.0)
        total_ssim += ssim_slice
    avg_ssim = total_ssim / ori_img.shape[2]
    return float(avg_ssim.numpy())


def MAE(ori_img, gene_img):
    """Compute the mean absolute error (MAE) per slice, averaged over all slices."""
    total_mae = 0
    for i in range(ori_img.shape[2]):  # ori_img.shape[2] = number of slices
        mae_per_slice = mean_absolute_error(ori_img[:, :, i], gene_img[:, :, i])
        total_mae += mae_per_slice
    avg_mae = total_mae / ori_img.shape[2]
    return avg_mae


def PSNR_slice_mean(ori_img, gene_img):
    """Compute the mean Peak Signal-to-Noise Ratio (PSNR) across slices."""
    total_psnr = 0
    for i in range(ori_img.shape[2]):
        psnr_i = psnr(ori_img[:, :, i], gene_img[:, :, i], data_range=255)
        total_psnr += psnr_i
    avg_psnr = total_psnr / ori_img.shape[2]
    return avg_psnr


def LPIPS(ori_img, gene_img):
    """Compute the average LPIPS perceptual distance between two 3D image volumes."""
    ori_img_conv = np.array(ori_img) / 127.5 - 1.  # convert from 0–255 to -1–1
    gene_img_conv = np.array(gene_img) / 127.5 - 1.
    loss_fn_alex = lpips.LPIPS(net='alex')
    total_lpips = 0
    for i in range(ori_img.shape[2]):
        ori_img_2D = np.expand_dims(np.stack([ori_img_conv[:, :, i]] * 3, 0), axis=0)  # (1, 3, H, W)
        gene_img_2D = np.expand_dims(np.stack([gene_img_conv[:, :, i]] * 3, 0), axis=0)
        ori_torch = torch.from_numpy(ori_img_2D).float()
        gene_torch = torch.from_numpy(gene_img_2D).float()
        lpips_per_slice = loss_fn_alex(ori_torch, gene_torch)
        total_lpips += lpips_per_slice
    avg_case_lpips = total_lpips / ori_img.shape[2]
    return avg_case_lpips.item()
