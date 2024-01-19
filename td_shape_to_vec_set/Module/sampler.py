import torch

from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond


class Sampler(object):
    def __init__(self) -> None:
        self.model = EDMPrecond(n_latents=512, channels=16)

        self.device = "cpu"
        return

    def sample(self) -> bool:
        self.model.eval()

        # self.model.load_state_dict(torch.load(self.dm_pth)["model"])
        # self.model.to(self.device)

        total = 1000
        iters = 100

        with torch.no_grad():
            for category_id in [18]:
                print(category_id)
                for i in range(total // iters):
                    print(
                        "start diffuse itr No."
                        + str(i + 1)
                        + "/"
                        + str(total // iters)
                        + "..."
                    )
                    sampled_array = self.model.sample(
                        cond=torch.Tensor([category_id] * iters).long().to(self.device),
                        batch_seeds=torch.arange(i * iters, (i + 1) * iters).to(
                            self.device
                        ),
                    ).float()

                    print(
                        sampled_array.shape,
                        sampled_array.max(),
                        sampled_array.min(),
                        sampled_array.mean(),
                        sampled_array.std(),
                    )

        return True
