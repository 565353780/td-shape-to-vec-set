import torch
from torch import nn


from td_shape_to_vec_set.Model.AutoEncoder.pre_norm import PreNorm
from td_shape_to_vec_set.Model.AutoEncoder.feed_forward import FeedForward
from td_shape_to_vec_set.Model.AutoEncoder.attention import Attention
from td_shape_to_vec_set.Model.AutoEncoder.point_embed import PointEmbed
from td_shape_to_vec_set.Method.io import exists
from td_shape_to_vec_set.Method.cache import cache_fn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=True,
    ):
        super().__init__()

        self.depth = depth

        self.point_embed = PointEmbed(dim=dim)

        def get_latent_attn():
            return PreNorm(
                dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
            )

        def get_latent_ff():
            return PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        )

    def decode(self, x, queries):
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents)

    def forward(self, pc, queries):
        x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        return {"logits": o}


def test():
    dim = 512

    model = AutoEncoder(
        depth=24,
        dim=dim,
        queries_dim=dim,
        output_dim=1,
        heads=8,
        dim_head=64,
    )

    torch.save(model.state_dict(), "./output/test.pth")
    return True
