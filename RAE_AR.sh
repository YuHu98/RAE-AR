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
