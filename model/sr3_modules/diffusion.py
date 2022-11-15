import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
class AFD(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels, att_f):
        super(AFD, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @autocast()
    def forward(self, fm_s, fm_t, eps=1e-6):
        fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
        rho = self.attention(fm_t_pooled)
        # rho = F.softmax(rho.squeeze(), dim=-1)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
def E_(input, t, shape):
    out = torch.gather(input, 0, t-1).repeat(shape[0], 1)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        noise,_,_=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        if condition_x is not None:
            x_recon= self.predict_start_from_noise(
                x, t=t, noise=noise)
        else:
            noise,_,_=self.denoise_fn(x, noise_level)
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None,t=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        if t==None:
            t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon,stem_small, stem_mid= self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon,stem_small, stem_mid = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        noise_recon = default(x_recon, lambda: torch.randn_like(x_start))
        x0_pred = (x_noisy - (
                1 - continuous_sqrt_alpha_cumprod.view(-1, 1, 1,
                                                       1) ** 2).sqrt() * noise_recon) / continuous_sqrt_alpha_cumprod.view(
            -1, 1, 1, 1)

        # loss = self.loss_func(noise, x_recon)
        return noise,x0_pred,t,stem_small, stem_mid,x_recon

    @torch.no_grad()
    def get_alpha_sigma(self, x, t):
        alpha = E_(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma
    def distill_loss(self, student_diffusion, x_in, tt,ts,  noise=None, gamma=0.3,student_device=None):
        x_0 = x_in['HR']
        tt = torch.tensor(tt).to(x_0 .device)
        ts = torch.tensor(ts).to(x_0 .device)

        [b, c, h, w] = x_0.shape
        continuous_sqrt_alpha_cumprod_st = torch.FloatTensor(
            np.random.uniform(
                student_diffusion.sqrt_alphas_cumprod_prev[ts - 1],
                student_diffusion.sqrt_alphas_cumprod_prev[ts],
                size=b
            )
        ).to(x_0.device)
        continuous_sqrt_alpha_cumprod_st = continuous_sqrt_alpha_cumprod_st.view(
            b, -1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[tt - 1],
                self.sqrt_alphas_cumprod_prev[tt],
                size=b
            )
        ).to(x_0.device)
        noise = default(noise, lambda: torch.randn_like(x_0))
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)
        z = self.q_sample(
            x_start=x_0, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise)

        with torch.no_grad():
            alpha_s, sigma_s = continuous_sqrt_alpha_cumprod_st.view(-1, 1, 1, 1), (
                        1 - continuous_sqrt_alpha_cumprod_st ** 2).sqrt().view(-1, 1, 1, 1)
            alpha_1, sigma_1 = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), (
                    1 - continuous_sqrt_alpha_cumprod ** 2).sqrt().view(-1, 1, 1, 1)
            v_1, stem_small_1, stem_mid_1 = self.denoise_fn(
                torch.cat([x_in['SR'], z], dim=1), continuous_sqrt_alpha_cumprod)
            x_2 = (1 / alpha_1 * (z - sigma_1 * v_1)).clip(-1, 1)

            v_2 = (z - alpha_s * x_2) / sigma_s
        v, stem_small_2, stem_mid_2 = student_diffusion.denoise_fn(torch.cat([x_in['SR'], z], dim=1),
                                                                   continuous_sqrt_alpha_cumprod_st)
        l_pix = self.loss_func(v, v_2).sum() / int(
            b * c * h * w)
        l_pix_st = self.loss_func(v, noise).sum() / int(
            b * c * h * w)

        return 0.5*l_pix+l_pix_st
    @autocast()
    def forward(self, x,  *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
