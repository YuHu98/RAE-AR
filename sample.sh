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