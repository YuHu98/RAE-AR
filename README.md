# RAE-AR: Taming Autoregressive Models with Representation Autoencoders <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2503.05305-b31b1b.svg)](https://arxiv.org/abs/2604.01545)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_demo-green)](https://yuhu98.github.io//projects/RAE-AR.html)&nbsp;
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-rae_ar-yellow)](https://huggingface.co/figereatfish/RAE-AR)&nbsp;


## 📰 News

- [2026-4-3] We release the code and checkpoint of `RAE-AR`.
- [2026-4-2] The [tech report](https://arxiv.org/abs/2604.01545) of `RAE-AR` is available.


## Preparation

### Installation

Download the code:
```
git clone https://github.com/YuHu98/RAE-AR.git
cd RAE-AR
```

Create environment:

```
conda create -n rae python=3.10 -y
conda activate rae
pip install uv

# Install PyTorch 2.8.0 with CUDA 12.9 # or your own cuda version
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 

# Install other dependencies
uv pip install -r requirements.txt
```

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in `/data/`.


### Pretrained Weights
Download pre-trained semantic encoders [DINOv2](https://huggingface.co/facebook/dinov2-with-registers-base/tree/main)/[SigLIP2](https://huggingface.co/google/siglip2-base-patch16-256)/[MAE](https://huggingface.co/facebook/vit-mae-base), and place them in `/models/`.

Download the weights for the decoder of [RAE](https://huggingface.co/nyu-visionx/RAE-collections/tree/main), and place it in `/models/`.

Download the weights of [VAE](https://huggingface.co/figereatfish/FAR/tree/main/vae)/[VAVAE](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/tree/main), and place them in `/vae_pth/`.

Download [.npz](https://huggingface.co/figereatfish/FAR/tree/main/fid_stats) of ImageNet 256x256 for calculating the FID metric, and place it in `/fid_stats/`.

For convenience, our pre-trained MAR models can be downloaded directly here as well:

| MAR Model                                                              | FID-50K | Inception Score |
|------------------------------------------------------------------------|---------|-----------------|
| [RAE-AR(DINOv2)](https://huggingface.co/figereatfish/RAE-AR)           | 7.494   | 165.588         |
| [RAE-AR(SigLIP2)](https://huggingface.co/figereatfish/RAE-AR)          | 6.091   | 241.768         |
| [RAE-AR(MAE)](https://huggingface.co/figereatfish/RAE-AR)              | 9.083   | 129.770         |



## Training

```
bash RAE_AR.sh
```

Specifically, take the default script for example:
```
data_path=./data/imagenet
result_path=./output/AR_DINOv2
ar_type=ar
vae_type=dinov2
rae_config=./configs/RAE_DINOv2.yaml
vae_path=./vae_pth/kl16.ckpt

torchrun  --nnodes=1  --nproc_per_node=8   --master_port=16385  \
train.py  --results-dir $result_path  --data-path $data_path  --image-size 256   --patch-size 16  --ema 0.9999  \
--model SphereAR-H  --epochs 300  --latent-dim 768   --lr 3e-4  --global-batch-size 1024   --no-compile  \
--ar-type $ar_type  --vae-type $vae_type  --rae-config $rae_config  --vae-path $vae_path  \
--noise-std 0.2  --use-token-norm  --enable-time-shift
```

- Add `--use-token-norm` to enable token-wise normalization for reducing token variance.
- Adjust `--noise-std` to control the intensity of noise injuction for reducing the exposure bias.
- Add `--enable-time-shift` to enable dimension-dependent schedule. This is disabled by default.
- Two autoregressve generation types: `ar_type` for `ar / mar`
- Five autoencoder types: `ar_type` for `kl_vae / vavae / dinov2 / siglip2 / mae`.
- Modify `rae_config or vae_path` to match the corresponding autoencoder type.
- Modify `--latent-dim` to match the corresponding autoencoder type. `kl_vae(16) / vavae(32) / dinov2(768) / siglip2(768) / mae(768)`.



## Evaluation
```
bash samle.sh
```

Specifically, take the default inference script for example:
```
ckpt=./output/AR_DINOv2/epoch_300.pt
result_path=./samples/AR_DINOv2/ep300
ar_type=ar
vae_type=dinov2
rae_config=./configs/RAE_DINOv2.yaml
vae_path=./vae_pth/kl16.ckpt

torchrun --nnodes=1 --nproc_per_node=8  --master_port=16384 \
sample_ddp.py  --sample-dir $result_path --ckpt $ckpt --cfg-scale 4.5  --patch-size 16 \
--model SphereAR-H  --latent-dim 768  --per-proc-batch-size 250  --to-npz  \
--ar-type $ar_type  --vae-type $vae_type  --rae-config $rae_config  --vae-path $vae_path \
```





## Acknowledgements

A large portion of codes in this repo is based on [RAE](https://github.com/bytetriper/RAE/tree/main), and [SphereAR](https://github.com/guolinke/SphereAR/tree/main). Thanks for these great work and open source。

## Contact

If you have any questions, feel free to contact me through email (yuhu520@mail.ustc.edu.cn). Enjoy!

## Citation
```BibTeX
@article{yu2026raear,
  author    = {Yu, Hu and Xu, Hang and Huang, Jie and Xue, Zeyue and Huang, Haoyang and Duan, Nan and Zhao, Feng},
  title     = {RAE-AR: Taming Autoregressive Models with Representation Autoencoders},
  journal   = {arxiv:2604.01545},
  year      = {2026}
}
```
