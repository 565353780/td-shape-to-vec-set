import torch
import torch.nn as nn

from td_shape_to_vec_set.Model.Transformer.latent_array import LatentArrayTransformer
from td_shape_to_vec_set.Method.sample import edm_sampler
from td_shape_to_vec_set.Module.stacked_random_generator import StackedRandomGenerator


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        n_latents=512,
        channels=8,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=12,
        context_dim=512,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.category_emb = nn.Embedding(55, context_dim)

        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )
        return

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forwardCondition(self, x, sigma, condition, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * x).to(dtype), c_noise.flatten(), cond=condition, **model_kwargs
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def forward(self, x, sigma, condition, force_fp32=False, **model_kwargs):
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([x.shape[0]], dtype=torch.long, device=x.device))
        else:
            condition = self.emb_category(condition)

        return self.forwardCondition(x, sigma, condition, force_fp32, **model_kwargs)

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_seeds=None,
        diffuse_steps:int = 18,
    ) -> list:
        if cond is not None:
            batch_size, device = *cond.shape, cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size)
        else:
            assert batch_seeds is not None
            device = batch_seeds.device
            batch_size = batch_seeds.shape[0]

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, self.n_latents, self.channels], device=device)

        return edm_sampler(self, latents, cond, randn_like=rnd.randn_like, num_steps=diffuse_steps)


def kl_d512_m512_l8_edm():
    return EDMPrecond(n_latents=512, channels=8)


def kl_d512_m512_l16_edm():
    return EDMPrecond(n_latents=512, channels=16)


def kl_d512_m512_l32_edm():
    return EDMPrecond(n_latents=512, channels=32)


def kl_d512_m512_l4_d24_edm():
    return EDMPrecond(n_latents=512, channels=4, depth=24)


def kl_d512_m512_l8_d24_edm():
    return EDMPrecond(n_latents=512, channels=8, depth=24)


def kl_d512_m512_l32_d24_edm():
    return EDMPrecond(n_latents=512, channels=32, depth=24)
