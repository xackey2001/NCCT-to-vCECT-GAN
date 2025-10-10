# NCCT-to-vCECT-GAN

This repository contains an implementation of a **Registration-Guided GAN (RegGAN)**  
for synthesizing virtual contrast-enhanced CT (vCECT) from non-contrast CT (NCCT).  
It also supports **Pix2Pix mode** by disabling the registration module.

---

## 🚀 Features
- 3D registration-guided GAN framework for thoracic imaging
- Optional Pix2Pix mode (`regist=False`)
- Supports DICOM loading, windowing, and 2D/3D input
- Training and inference scripts included

```text
NCCT-to-vCECT-GAN/
├─ examples/
│   ├─ train_reggan.py
│   └─ inference_reggan.py
├─ trainer/
│   ├─ reg_gan_trainer.py
│   ├─ reg_gan_reg_unified.py
│   ├─ reg_gan_layers_unified.py
│   └─ reg_gan_transformer_unified.py
├─ Model/
│   └─ reg_gan_Models_torch.py
├─ reg_gan_dataloader.py
├─ reg_gan_metrics.py
└─ README.md
```

## 📁 Dataset Structure

Before training, organize your DICOM data in the following folder structure:
```
/dataset/
   ├─ case0001/
   │   ├─ CT1/ ← non-contrast CT (NCCT)
   │   │   ├─ 0001.dcm
   │   │   ├─ 0002.dcm
   │   │   └─ ...
   │   └─ CT2/ ← contrast-enhanced CT (CECT)
   │       ├─ 0001.dcm
   │       ├─ 0002.dcm
   │       └─ ...
   ├─ case0002/
   │   ├─ CT1/ ← non-contrast CT (NCCT)
   │   │   ├─ 0001.dcm
   │   │   ├─ 0002.dcm
   │   │   └─ ...
   │   └─ CT2/ ← contrast-enhanced CT (CECT)
   │       ├─ 0001.dcm
   │       ├─ 0002.dcm
   │       └─ ...
   ├─ case0003/
   │   ├─ CT1/ ← non-contrast CT (NCCT)
   │   └─ CT2/ ← contrast-enhanced CT (CECT)
   └─ ...
```

## 🧾 Data Preparation

If `select.ftr` (the dataset metadata file) has not been created yet, run the following script to generate it.  
This script loads DICOM metadata, converts slice positions to numeric values, sorts by patient ID and slice order, and saves the full dataset information as a `.ftr` file.

```python
cts = ('CT1', 'CT2')
df = my_dicoms_to_dataframe(traindir, cts)  # headers.ftr will be saved in 'traindir'.
df['zpos'] = df['zpos'].apply(pd.to_numeric)  # Convert 'zpos' from string to numeric.
df = df.sort_values(by=['pid', 'ct', 'zpos'])  # Sort by patient ID, CT type, and z-position.
df2 = df.reset_index(drop=True)  # Reset index after sorting.
df2path = os.path.join(spath, 'select2.ftr')
df2.to_feather(df2path)  # Save the dataset metadata to 'spath'.
```

```
#Example configuration inside the script:
cfg = {
    'df_path': os.path.join(spath, 'select.ftr'),
    'cts': ('CT1', 'CT2'),
    'img_shape': (256,256,16),
    'window1': [(500,50)],
    'window2': [(500,50)],
    'lrs': (0.0001, 0.1),
    'L_weights': (1,100),
    'grid': (4,4,4),
    'regist': True,  # False enables Pix2Pix mode (direct NCCT-to-vCECT)
    'randomshift': 0.1
}
```


## 🏋️ Training
python examples/train_reggan.py

## ✅Inference
python examples/inference_reggan.py




