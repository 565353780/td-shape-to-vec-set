from td_shape_to_vec_set.Dataset.axis_scaling import AxisScaling
from td_shape_to_vec_set.Dataset.shapenet import ShapeNet


def build_shape_surface_occupancy_dataset(split, args):
    if split == "train":
        # transform = #transforms.Compose([
        transform = AxisScaling((0.75, 1.25), True)
        # ])
        return ShapeNet(
            args.data_path,
            split=split,
            transform=transform,
            sampling=True,
            num_samples=1024,
            return_surface=True,
            surface_sampling=True,
            pc_size=args.point_cloud_size,
        )
    elif split == "val":
        # return ShapeNet(args.data_path, split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
        return ShapeNet(
            args.data_path,
            split=split,
            transform=None,
            sampling=False,
            return_surface=True,
            surface_sampling=True,
            pc_size=args.point_cloud_size,
        )
    else:
        return ShapeNet(
            args.data_path,
            split=split,
            transform=None,
            sampling=False,
            return_surface=True,
            surface_sampling=True,
            pc_size=args.point_cloud_size,
        )


if __name__ == "__main__":
    # m = ShapeNet('/home/zhanb0b/data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
    m = ShapeNet(
        "/home/zhanb0b/data/",
        "train",
        transform=AxisScaling(),
        sampling=True,
        num_samples=1024,
        return_surface=True,
        surface_sampling=True,
    )
    p, l, s, c = m[0]
    print(p.shape, l.shape, s.shape, c)
    print(p.max(dim=0)[0], p.min(dim=0)[0])
    print(p[l == 1].max(axis=0)[0], p[l == 1].min(axis=0)[0])
    print(s.max(axis=0)[0], s.min(axis=0)[0])
