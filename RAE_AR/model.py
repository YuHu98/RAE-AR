import argparse
from functools import partial
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from .diff_head import DiffHead
from .layers import TransformerBlock, get_2d_pos, precompute_freqs_cis_2d
from .vae import VAE
from .KL_vae import AutoencoderKL
from .vavae import AutoencoderKL_VAVAE
from .rae_utils import parse_configs, instantiate_from_config
import scipy.stats as stats
import torch.nn.init as init

def get_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(SphereAR_models.keys()), default="SphereAR-L"
    )
    parser.add_argument("--vae-only", action="store_true", help="only train vae")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--patch-size", type=int, default=16, choices=[16,32])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--diff-batch-mul", type=int, default=4)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--enable-time-shift",action="store_true")
    return parser

    
def create_model(args, device):
    model = SphereAR_models[args.model](
        resolution=args.image_size,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        vae_only=args.vae_only,
        diff_batch_mul=args.diff_batch_mul,
        cls_token_num=args.cls_token_num,
        num_classes=args.num_classes,
        grad_checkpointing=args.grad_checkpointing,
        vae_type=args.vae_type,
        rae_config=args.rae_config,
        ar_type=args.ar_type,
        use_token_norm = args.use_token_norm,
        vae_path = args.vae_path,
        enable_time_shift = args.enable_time_shift,
    ).to(device, memory_format=torch.channels_last)
    return model


from torchvision.utils import make_grid
from typing import Optional
from PIL import Image
def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None, format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


    
class SphereAR(nn.Module):

    def __init__(
        self,
        dim,
        n_layer,
        n_head,
        diff_layers,
        diff_dim,
        diff_adanln_layers,
        latent_dim,
        patch_size,
        resolution,
        vae_type,
        rae_config,
        ar_type,
        diff_batch_mul,
        vae_path,
        vae_only=False,
        grad_checkpointing=False,
        cls_token_num=16,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        use_token_norm: bool = False,
        enable_time_shift: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_layer = n_layer
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.cls_token_num = cls_token_num
        self.class_dropout_prob = class_dropout_prob
        self.latent_dim = latent_dim

        self.use_token_norm = use_token_norm
        self.vae_path = vae_path
        self.noise_std = 0.0
        self.enable_time_shift = enable_time_shift

        self.vae_type = vae_type
        self.ar_type = ar_type
        if vae_type=='s_vae':
            self.vae = VAE(latent_dim=latent_dim, image_size=resolution, patch_size=patch_size)
        elif vae_type=='kl_vae':
            self.vae = AutoencoderKL(embed_dim=latent_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=vae_path)
        elif vae_type=='vavae':
            self.vae = AutoencoderKL_VAVAE(embed_dim=latent_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=vae_path)
        elif vae_type=='dinov2' or vae_type=='siglip2' or vae_type=='mae':
            rae_conf = parse_configs(rae_config)
            
            if "params" not in rae_conf:
                rae_conf["params"] = dict()
            rae_conf["params"]["use_token_norm"] = self.use_token_norm
            
            self.vae = instantiate_from_config(rae_conf)


        else:
            raise ValueError(f"Unsupported vae_type: '{vae_type}'. Supported types are: ['s_vae', 'kl_vae', 'vavae', 'dinov2', 'siglip2', 'mae']")

        self.vae_only = vae_only
        self.grad_checkpointing = grad_checkpointing
        self.mask_ratio_generator = stats.truncnorm((0.7 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        if not vae_only:
            self.cls_embedding = nn.Embedding(num_classes + 1, dim * self.cls_token_num)
            self.proj_in = nn.Linear(latent_dim, dim, bias=True)
            self.emb_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
            self.h, self.w = resolution // patch_size, resolution // patch_size
            self.seq_len = self.h * self.w
            self.total_tokens = self.h * self.w + self.cls_token_num

            self.layers = torch.nn.ModuleList()
            if self.ar_type=='ar':
                for layer_id in range(n_layer):
                    self.layers.append(
                        TransformerBlock(
                            dim,
                            n_head,
                            causal=True,
                        )
                    )
            elif self.ar_type=='mar':
                for layer_id in range(n_layer):
                    self.layers.append(
                        TransformerBlock(
                            dim,
                            n_head,
                            causal=False,
                        )
                    )
            else:
                raise ValueError(f"Unsupported ar_type: '{ar_type}'. Supported types are: ['ar', 'mar']")

            self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
            self.pos_for_diff = nn.Embedding(self.h * self.w, dim)
            self.head = DiffHead(
                ch_target=latent_dim,
                ch_cond=dim,
                ch_latent=diff_dim,
                depth_latent=diff_layers,
                depth_adanln=diff_adanln_layers,
                grad_checkpointing=grad_checkpointing,
            )
            self.diff_batch_mul = diff_batch_mul

            patch_2d_pos = get_2d_pos(resolution, patch_size)

            if self.ar_type=='ar':
                self.register_buffer(
                    "freqs_cis",
                    precompute_freqs_cis_2d(
                        patch_2d_pos,
                        dim // n_head,
                        10000,
                        cls_token_num=self.cls_token_num,
                    )[:-1],
                    persistent=False,
                )
            elif self.ar_type=='mar':
                self.register_buffer(
                    "freqs_cis",
                    precompute_freqs_cis_2d(
                        patch_2d_pos,
                        dim // n_head,
                        10000,
                        cls_token_num=self.cls_token_num,
                    ),
                    persistent=False,
                )
            self.freeze_vae()

        self.initialize_weights()

    def non_decay_keys(self):
        return ["proj_in", "cls_embedding"]

    def freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def freeze_vae(self):
        self.freeze_module(self.vae)
        self.vae.eval()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        # self.apply(self.__init_weights)
        if not self.vae_only:
            self.head.initialize_weights()
        # self.vae.initialize_weights()

    def __init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def drop_label(self, class_id):
        if self.class_dropout_prob > 0.0 and self.training:
            is_drop = (
                torch.rand(class_id.shape, device=class_id.device)
                < self.class_dropout_prob
            )
            class_id = torch.where(is_drop, self.num_classes, class_id)
        return class_id


    def generate_random_mask(self, x):
        bsz, seq_len, _ = x.shape
        device = x.device

        noise = torch.rand(bsz, seq_len, device=device)
        orders = torch.argsort(noise, dim=1)
        temp_tensor = torch.empty(1)
        init.trunc_normal_(temp_tensor, mean=1.0, std=0.25, a=0.7, b=1.0)
        mask_ratio = temp_tensor
        num_masked_tokens = int(np.ceil(seq_len * mask_ratio))
        mask = torch.zeros(bsz, seq_len, device=device)
        mask_indices = orders[:, :num_masked_tokens]
        mask.scatter_(dim=1, index=mask_indices, value=1.0)
        return mask


    def sample_orders(self, x):
        bsz, seq_len, _ = x.shape
        device = x.device
        noise = torch.rand(bsz, seq_len, device=device)
        orders = torch.argsort(noise, dim=1)
        return orders

    def forward(
        self,
        images,
        class_id,
    ):

        if self.vae_type=='s_vae':
            vae_latent, kl_loss = self.vae.encode(images)
        elif self.vae_type=='kl_vae':
            vae_latent = self.vae.encode(images).sample().mul_(0.2325)
            vae_latent = vae_latent.permute(0, 2, 3, 1).flatten(1, 2)
            kl_loss = 0
        elif self.vae_type=='vavae':
            vae_latent = self.vae.encode(images).sample()
            vae_latent = vae_latent.permute(0, 2, 3, 1).flatten(1, 2)
            kl_loss = 0
        elif self.vae_type=='dinov2' or self.vae_type=='siglip2' or self.vae_type=='mae':
            vae_latent = self.vae.encode(images)

            vae_latent = vae_latent.permute(0, 2, 3, 1).flatten(1, 2)
            kl_loss = 0
        
        if not self.vae_only:
            x = vae_latent.detach()


            if self.training:

                # --- 2. 注入噪声 ---
                # 注意：当 x 的标准差被固定为 1.0 时，0.5 的噪声是非常大的（信噪比 2:1）
                # 如果发现模型收敛困难，可以考虑降低到 0.05 或 0.1
                noise = torch.randn_like(x) * self.noise_std
                noise = noise.to(x.device)
                x = x + noise
            

            if self.ar_type=='mar':
                x = self.proj_in(x)
                mask = self.generate_random_mask(x)
                x = x * (1 - mask).unsqueeze(-1)

            elif self.ar_type=='ar':
                x = self.proj_in(x[:, :-1, :])

            class_id = self.drop_label(class_id)
            bsz = x.shape[0]
            c = self.cls_embedding(class_id).view(bsz, self.cls_token_num, -1)
            x = torch.cat([c, x], dim=1)
            x = self.emb_norm(x)

            if self.grad_checkpointing and self.training:
                for layer in self.layers:
                    block = partial(layer.forward, freqs_cis=self.freqs_cis)
                    x = checkpoint(block, x, use_reentrant=False)
            else:
                for layer in self.layers:
                    x = layer(x, self.freqs_cis)

            x = x[:, -self.h * self.w :, :]
            x = self.norm(x)
            x = x + self.pos_for_diff.weight

            target = vae_latent.detach()
            x = x.view(-1, x.shape[-1])
            target = target.reshape(-1, target.shape[-1])

            x = x.repeat(self.diff_batch_mul, 1)
            target = target.repeat(self.diff_batch_mul, 1)
            if self.enable_time_shift:
                alpha = (self.latent_dim * (self.resolution // self.patch_size) * (self.resolution // self.patch_size) / 4096)**0.5
            else:
                alpha = 1.0

            if self.ar_type=='mar':
                mask = mask.reshape(bsz*self.seq_len).repeat(self.diff_batch_mul)
                loss = self.head(target, x, mask = mask, alpha = alpha)
            elif self.ar_type=='ar':
                loss = self.head(target, x, mask = None, alpha = alpha)
            recon = None
        else:
            loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
            recon = self.vae.decode(vae_latent)

        return loss, kl_loss, recon

    def enable_kv_cache(self, bsz):
        for layer in self.layers:
            layer.attention.enable_kv_cache(bsz, self.total_tokens)

    @torch.compile()
    def forward_model(self, x, start_pos, end_pos):
        x = self.emb_norm(x)
        for layer in self.layers:
            x = layer.forward_onestep(x, self.freqs_cis[start_pos:end_pos,], start_pos, end_pos)
        x = self.norm(x)
        return x

    @torch.compile()
    def forward_model_mar(self, x):
        x = self.emb_norm(x)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = x[:, -self.h * self.w :, :]
        x = self.norm(x)
        return x

    def head_sample(self, x, diff_pos, sample_steps, cfg_scale, cfg_schedule="linear", alpha = 1.0):
        x = x + self.pos_for_diff.weight[diff_pos : diff_pos + 1, :]
        x = x.view(-1, x.shape[-1])
        seq_len = self.h * self.w
        if cfg_scale > 1.0:
            if cfg_schedule == "constant":
                cfg_iter = cfg_scale
            elif cfg_schedule == "linear":
                start = 1.0
                cfg_iter = start + (cfg_scale - start) * diff_pos / seq_len
            else:
                raise NotImplementedError(f"unknown cfg_schedule {cfg_schedule}")
        else:
            cfg_iter = 1.0
        pred = self.head.sample(x, num_sampling_steps=sample_steps, cfg=cfg_iter, alpha=alpha)
        pred = pred.view(-1, 1, pred.shape[-1])
        # Important: normalize here, for both next-token prediction and vae decoding
        if self.vae_type=='s_vae':  # or self.vae_type=='dinov2':
            pred = self.vae.normalize(pred)
        return pred

    def head_sample_mar(self, x, sample_steps, cfg_iter,alpha = 1.0):
        x = x.view(-1, x.shape[-1])
        pred = self.head.sample(x, num_sampling_steps=sample_steps, cfg=cfg_iter, alpha=alpha)
        if self.vae_type=='s_vae':
            pred = self.vae.normalize(pred)
        return pred

    def mask_by_order(self, mask_len, order, bsz, seq_len):
        masking = torch.zeros(bsz, seq_len).cuda()
        masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
        return masking

    
    @torch.no_grad()
    def sample(self, cond, sample_steps, cfg_scale=1.0, cfg_schedule="linear"):
        self.eval()
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        bsz = cond_combined.shape[0]
        act_bsz = bsz // 2 if cfg_scale > 1.0 else bsz
        self.enable_kv_cache(bsz)

        c = self.cls_embedding(cond_combined).view(bsz, self.cls_token_num, -1)
        last_pred = None
        all_preds = []
        if self.enable_time_shift:
            alpha = (self.latent_dim * (self.resolution // self.patch_size) * (self.resolution // self.patch_size) / 4096)**0.5
        else:
            alpha = 1.0

        for i in range(self.h * self.w):
            if i == 0:
                x = self.forward_model(c, 0, self.cls_token_num)
            else:
                x = self.proj_in(last_pred)
                x = self.forward_model(x, i + self.cls_token_num - 1, i + self.cls_token_num)
            last_pred = self.head_sample(
                x[:, -1:, :],
                i,
                sample_steps,
                cfg_scale,
                cfg_schedule,
                alpha=alpha,
            )
            all_preds.append(last_pred)

        x = torch.cat(all_preds, dim=-2)[:act_bsz]

        if self.vae_type=='s_vae':
            recon = self.vae.decode(x)
        elif self.vae_type=='kl_vae':
            B_, L_, C_ = x.shape
            H_ = W_ = int(math.isqrt(L_)) # isqrt 是整数平方根
            if H_ * W_ != L_:
                raise ValueError(f"Sequence length {L_} is not a perfect square, cannot auto-reshape.")
            x = x.transpose(1, 2).reshape(B_, C_, H_, W_)
            recon = self.vae.decode(x / 0.2325)
        elif self.vae_type=='vavae':
            B_, L_, C_ = x.shape
            H_ = W_ = int(math.isqrt(L_)) # isqrt 是整数平方根
            if H_ * W_ != L_:
                raise ValueError(f"Sequence length {L_} is not a perfect square, cannot auto-reshape.")
            x = x.transpose(1, 2).reshape(B_, C_, H_, W_)
            recon = self.vae.decode(x)
        elif self.vae_type=='dinov2' or self.vae_type=='siglip2' or self.vae_type=='mae':
            B_, L_, C_ = x.shape
            H_ = W_ = int(math.isqrt(L_)) # isqrt 是整数平方根
            if H_ * W_ != L_:
                raise ValueError(f"Sequence length {L_} is not a perfect square, cannot auto-reshape.")
            x = x.transpose(1, 2).reshape(B_, C_, H_, W_)
            recon = self.vae.decode(x)
        return recon
    


    @torch.no_grad()
    def sample_mar(self, cond, sample_steps, cfg_scale=1.0, cfg_schedule="linear"):
        self.eval()
        mask = torch.ones(cond.shape[0], self.h*self.w).cuda()
        x = torch.zeros(cond.shape[0], self.h*self.w, self.latent_dim).cuda()
        orders = self.sample_orders(x)

        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        bsz = cond_combined.shape[0]
        act_bsz = bsz // 2 if cfg_scale > 1.0 else bsz
        c = self.cls_embedding(cond_combined).view(bsz, self.cls_token_num, -1)

        last_pred = None
        num_iter = 64
        for step in range(num_iter):
            cur_x = x.clone()
            if cfg_scale > 1.0:
                x = torch.cat([x, x], dim=0)
                mask = torch.cat([mask, mask], dim=0)
            
            x = self.proj_in(x)
            x = x * (1 - mask.unsqueeze(-1).to(x.dtype))

            x = torch.cat([c, x], dim=1)
            x = self.forward_model_mar(x)
            x = x + self.pos_for_diff.weight


            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.h*self.w * mask_ratio)]).cuda()
            mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
            mask_next = self.mask_by_order(mask_len[0], orders, act_bsz, self.h*self.w)
            if step >= num_iter - 1:
                mask_to_pred = mask[:act_bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:act_bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg_scale == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg_scale - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg_scale
            else:
                raise NotImplementedError(f"unknown cfg_schedule {cfg_schedule}")

            x = x[mask_to_pred.nonzero(as_tuple=True)]


            if self.enable_time_shift:
                alpha = (self.latent_dim * (self.resolution // self.patch_size) * (self.resolution // self.patch_size) / 4096)**0.5
            else:
                alpha = 1.0
            
            last_pred = self.head_sample_mar(
                x,
                sample_steps,
                cfg_iter,
                alpha = alpha,
            )
            if not cfg_scale == 1.0:
                last_pred, _ = last_pred.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_x[mask_to_pred.nonzero(as_tuple=True)] = last_pred
            x = cur_x.clone()

        if self.vae_type=='s_vae':
            recon = self.vae.decode(x)
        elif self.vae_type=='kl_vae':
            B_, L_, C_ = x.shape
            H_ = W_ = int(math.isqrt(L_)) # isqrt 是整数平方根
            if H_ * W_ != L_:
                raise ValueError(f"Sequence length {L_} is not a perfect square, cannot auto-reshape.")
            x = x.transpose(1, 2).reshape(B_, C_, H_, W_)
            recon = self.vae.decode(x / 0.2325)
        elif self.vae_type=='vavae':
            B_, L_, C_ = x.shape
            H_ = W_ = int(math.isqrt(L_)) # isqrt 是整数平方根
            if H_ * W_ != L_:
                raise ValueError(f"Sequence length {L_} is not a perfect square, cannot auto-reshape.")
            x = x.transpose(1, 2).reshape(B_, C_, H_, W_)
            recon = self.vae.decode(x)
        elif self.vae_type=='dinov2' or self.vae_type=='siglip2' or self.vae_type=='mae':
            B_, L_, C_ = x.shape
            H_ = W_ = int(math.isqrt(L_)) # isqrt 是整数平方根
            if H_ * W_ != L_:
                raise ValueError(f"Sequence length {L_} is not a perfect square, cannot auto-reshape.")
            x = x.transpose(1, 2).reshape(B_, C_, H_, W_)
            recon = self.vae.decode(x)
            
        return recon

    def get_fsdp_wrap_module_list(self):
        return list(self.layers)


def SphereAR_H(**kwargs):
    return SphereAR(
        n_layer=40,
        n_head=20,
        dim=1280,
        diff_layers=3,
        diff_dim=1280,
        diff_adanln_layers=3,
        **kwargs,
    )

def SphereAR_L(**kwargs):
    return SphereAR(
        n_layer=32,
        n_head=16,
        dim=1024,
        diff_layers=8,
        diff_dim=1024,
        diff_adanln_layers=2,
        **kwargs,
    )


def SphereAR_B(**kwargs):
    return SphereAR(
        n_layer=24,
        n_head=12,
        dim=768,
        diff_layers=6,
        diff_dim=768,
        diff_adanln_layers=2,
        **kwargs,
    )


SphereAR_models = {
    "SphereAR-B": SphereAR_B,
    "SphereAR-L": SphereAR_L,
    "SphereAR-H": SphereAR_H,
}
