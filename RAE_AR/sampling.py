import torch


def get_score_from_velocity(velocity, x, t):
    alpha_t, d_alpha_t = t, 1
    sigma_t, d_sigma_t = 1 - t, -1
    mean = x
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * velocity - mean) / var
    return score


def get_velocity_from_cfg(velocity, cfg, cfg_mult):
    if cfg_mult == 2:
        cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
        velocity = uncond_v + cfg * (cond_v - uncond_v)
    return velocity


@torch.compile()
def euler_step(x, v, dt: float, cfg: float, cfg_mult: int):
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        v = get_velocity_from_cfg(v, cfg, cfg_mult)
        x = x + v * dt
    return x


@torch.compile()
def euler_maruyama_step(x, v, t, dt: float, cfg: float, cfg_mult: int):
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        v = get_velocity_from_cfg(v, cfg, cfg_mult)
        score = get_score_from_velocity(v, x, t)
        
        drift = v + (1 - t) * score
        
        noise_scale = (2.0 * (1.0 - t) * dt) ** 0.5
        
        x = x + drift * dt + noise_scale * torch.randn_like(x)
    return x


def euler_maruyama(
    input_dim,
    forward_fn,
    c: torch.Tensor,
    cfg: float = 1.0,
    num_sampling_steps: int = 20,
    last_step_size: float = 0.04,
    alpha: float = 1.0,
):
    cfg_mult = 1 if cfg <= 1.0 else 2
    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device)

    # 1. 生成原始线性序列 (0 -> 1.0 - last_step_size)
    # 包含起始点共 num_sampling_steps + 1 个点
    t_max_linear = 1.0 - last_step_size
    t_seq = torch.linspace(0, t_max_linear, num_sampling_steps + 1, device=c.device)

    # 2. 应用 Flow Shift 变换 (针对 0=噪声, 1=数据)
    # 直接在全局尺度上变换，让 alpha 决定步长的缩放
    t_seq_inv = 1.0 - t_seq
    t_seq_inv = (alpha * t_seq_inv) / (1 + (alpha - 1) * t_seq_inv)
    t_seq = 1.0 - t_seq_inv 

    t_batch = torch.zeros(c.shape[0], device=c.device)

    # 3. 循环采样 (前 num_sampling_steps 步)
    for i in range(num_sampling_steps):
        t_curr = t_seq[i]
        t_next = t_seq[i+1]
        dt = t_next - t_curr  # 这里的 dt 是经过 shift 缩放后的
        
        t_batch[:] = t_curr
        combined = torch.cat([x] * cfg_mult, dim=0)
        v = forward_fn(combined, t_batch, c)
        
        x = euler_maruyama_step(x, v, t_curr, dt, cfg, cfg_mult)

    # 4. 最后一步处理
    # 直接使用循环结束后的当前时间点 t_seq[-1] 作为起点
    # 终点固定为 1.0，步长即为剩下的距离
    t_final_start = t_seq[-1]
    final_dt = 1.0 - t_final_start 
    
    t_batch[:] = t_final_start
    combined = torch.cat([x] * cfg_mult, dim=0)
    v = forward_fn(combined, t_batch, c)
    
    x = euler_step(x, v, final_dt, cfg, cfg_mult)

    return torch.cat([x] * cfg_mult, dim=0)


def euler(
    input_dim,
    forward_fn,
    c,
    cfg: float = 1.0,
    num_sampling_steps: int = 50,
):
    cfg_mult = 1
    if cfg > 1.0:
        cfg_mult = 2

    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device)
    dt = 1.0 / num_sampling_steps
    t = 0
    t_batch = torch.zeros(c.shape[0], device=c.device)
    for _ in range(num_sampling_steps):
        t_batch[:] = t
        combined = torch.cat([x] * cfg_mult, dim=0)
        v = forward_fn(combined, t_batch, c)
        x = euler_step(x, v, dt, cfg, cfg_mult)
        t += dt

    return torch.cat([x] * cfg_mult, dim=0)
