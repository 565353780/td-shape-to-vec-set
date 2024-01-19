import torch


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, inputs, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([inputs.shape[0], 1, 1], device=inputs.device)
        # rnd_normal = torch.randn([1, 1, 1], device=inputs.device).repeat(inputs.shape[0], 1, 1)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(inputs) if augment_pipe is not None else (inputs, None)
        )

        n = torch.randn_like(y) * sigma

        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()
