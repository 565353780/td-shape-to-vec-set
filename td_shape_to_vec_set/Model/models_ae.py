from td_shape_to_vec_set.Model.auto_encoder import AutoEncoder
from td_shape_to_vec_set.Model.kl_auto_encoder import KLAutoEncoder


def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False):
    if determinisitc:
        model = AutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            heads=8,
            dim_head=64,
        )
    else:
        model = KLAutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
        )
    return model


def kl_d512_m512_l512(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=512, N=N, determinisitc=False)


def kl_d512_m512_l64(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=64, N=N, determinisitc=False)


def kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False)


def kl_d512_m512_l16(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=16, N=N, determinisitc=False)


def kl_d512_m512_l8(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, determinisitc=False)


def kl_d512_m512_l4(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=4, N=N, determinisitc=False)


def kl_d512_m512_l2(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=2, N=N, determinisitc=False)


def kl_d512_m512_l1(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=1, N=N, determinisitc=False)


###


def ae_d512_m512(N=2048):
    return create_autoencoder(dim=512, M=512, N=N, determinisitc=True)


def ae_d512_m256(N=2048):
    return create_autoencoder(dim=512, M=256, N=N, determinisitc=True)


def ae_d512_m128(N=2048):
    return create_autoencoder(dim=512, M=128, N=N, determinisitc=True)


def ae_d512_m64(N=2048):
    return create_autoencoder(dim=512, M=64, N=N, determinisitc=True)


###


def ae_d256_m512(N=2048):
    return create_autoencoder(dim=256, M=512, N=N, determinisitc=True)


def ae_d128_m512(N=2048):
    return create_autoencoder(dim=128, M=512, N=N, determinisitc=True)


def ae_d64_m512(N=2048):
    return create_autoencoder(dim=64, M=512, N=N, determinisitc=True)
