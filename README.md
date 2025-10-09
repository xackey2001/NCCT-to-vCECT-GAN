# NCCT-to-vCECT-GAN

This repository contains an implementation of a **Registration-Guided GAN (RegGAN)**  
for synthesizing virtual contrast-enhanced CT (vCECT) from non-contrast CT (NCCT).  
It also supports **Pix2Pix mode** by disabling the registration module.

---

## 🚀 Features
- 3D registration-guided GAN framework for thoracic imaging
- Optional Pix2Pix mode (`regist=False`)
- Supports DICOM loading, windowing, and resampling
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




